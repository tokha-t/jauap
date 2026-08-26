"""JAUAP Streamlit operator console. Run with: streamlit run app.py"""

from __future__ import annotations

import html
import hashlib
import io
import json
import math
import os
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from jauap.classify import FALLBACK_NOTICE
from jauap.deadline_engine import (
    extension_deadline,
    legal_tooltip,
    notification_deadline,
    working_days_between,
)
from jauap.draft import DRAFT_BANNER, generate_cluster_notifications
from jauap.pipeline import process_records


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEMO_RESULTS = DATA_DIR / "demo_results.json"
DEMO_SCORE = DATA_DIR / "demo_score.json"
ROUTING = json.loads((DATA_DIR / "routing.json").read_text(encoding="utf-8"))

RISK_LABELS = {"green": "🟢 Низкий", "amber": "🟠 Средний", "red": "🔴 Высокий"}
RISK_COLORS = {"green": "#18864B", "amber": "#C67605", "red": "#C92A2A"}
ENFORCEMENT_HELP = (
    "КоАП ст. 189 исключена с 1 июля 2021 года: административного штрафа за пропуск срока нет. "
    "Вместо этого Закон «О государственной службе» ст. 50(1)(7), 50(1)(11), 44(5) предусматривает "
    "дисциплинарный трек вплоть до увольнения при повторном нарушении."
)


st.set_page_config(page_title="JAUAP · диспетчеризация обращений", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
      :root { --j-navy:#17324d; --j-teal:#087f78; --j-line:#d9e2e8; --j-soft:#f4f7f8; }
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px; }
      h1, h2, h3 { color: var(--j-navy); letter-spacing: -0.015em; }
      .demo-banner { border-left: .35rem solid #d97706; background:#fff7e6; color:#593b05;
        padding:.75rem 1rem; margin:.5rem 0 1rem; font-weight:700; }
      .draft-banner { border-left:.35rem solid #b42318; background:#fff3f2; color:#7a271a;
        padding:.55rem .8rem; margin:.5rem 0; font-weight:700; }
      .metric-hero { border-top:.35rem solid var(--j-teal); background:var(--j-soft); padding:1rem 1.25rem;
        margin:.25rem 0 1rem; }
      .metric-hero .value { color:var(--j-navy); font-size:2.5rem; font-weight:750; line-height:1.1; }
      .metric-hero .label { color:#435466; font-size:.9rem; margin-top:.35rem; }
      .map-caption { border:1px solid var(--j-line); background:var(--j-soft); padding:.65rem .8rem;
        color:#435466; font-size:.9rem; }
      mark { background:#ffe08a; color:#2b2b2b; padding:.05rem .15rem; }
      div[data-testid="stMetric"] { border-top: .2rem solid var(--j-line); padding-top:.7rem; }
      @media (max-width: 768px) { .block-container { padding-left:1rem; padding-right:1rem; }
        .metric-hero .value { font-size:2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _embed_leaflet_assets(fmap: folium.Map) -> None:
    """Replace Folium's CDN defaults with same-origin committed assets."""
    fmap.default_js = [
        ("leaflet", "/app/static/leaflet.js"),
        ("jquery", "/app/static/jquery.min.js"),
    ]
    fmap.default_css = [("leaflet_css", "/app/static/leaflet.css")]


def _load_frozen_demo() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if DEMO_RESULTS.exists():
        payload = json.loads(DEMO_RESULTS.read_text(encoding="utf-8"))
        if payload.get("classification_source") != "Anthropic API via classify_text":
            raise ValueError("Legacy demo results are not model-produced")
        return payload["cases"], payload["clusters"], []
    raise FileNotFoundError("Run scripts/freeze_demo.py before loading the offline demo")


def _parse_upload(upload: Any) -> list[str]:
    if upload is None:
        return []
    raw = upload.getvalue()
    suffix = Path(upload.name).suffix.casefold()
    if suffix == ".txt":
        return [line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(raw))
        column = "raw_text" if "raw_text" in frame.columns else frame.columns[0]
        return [str(value).strip() for value in frame[column].dropna() if str(value).strip()]
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("JSON должен содержать список обращений")
    return [str(item.get("raw_text", "")).strip() if isinstance(item, dict) else str(item).strip() for item in payload]


def _live_records(texts: list[str]) -> list[dict[str, Any]]:
    now = datetime.now().replace(microsecond=0).isoformat()
    return [{
        "id": f"LIVE-{index:03d}", "raw_text": text, "received_at": now,
        "channel": "ввод оператора", "applicant_name": f"Синтетический заявитель {index}",
        "language_detected": "ru", "synthetic": True,
    } for index, text in enumerate(texts, start=1)]


def _deadline_color(value: int) -> str:
    if value < 0:
        return "background-color:#fde8e7;color:#8a1c13;font-weight:700"
    if value <= 3:
        return "background-color:#fff1d6;color:#754400;font-weight:700"
    return "background-color:#eaf7ef;color:#126b3a"


def _highlighted_text(case: dict[str, Any]) -> str:
    escaped = html.escape(case["raw_text"])
    mention = (case.get("location") or {}).get("raw_mention")
    if mention:
        escaped_mention = html.escape(mention)
        escaped = escaped.replace(escaped_mention, f"<mark>{escaped_mention}</mark>", 1)
    return escaped


def _case_detail(case: dict[str, Any], clusters: list[dict[str, Any]]) -> None:
    st.subheader(f"Дело {case['id']}")
    if case["urgency"] == "emergency":
        st.error("НЕМЕДЛЕННАЯ ПЕРЕДАЧА — АППК ст. 64(7-2)")
    st.markdown(f"<p>{_highlighted_text(case)}</p>", unsafe_allow_html=True)
    st.caption(f"Канал: {case['channel']} · Язык: {case['language_detected']} · Кластер: {case['cluster_id']}")
    st.info(
        "Почему не регулярные выражения: различие между заявлением, жалобой и сообщением меняет срок, "
        "процедуру и обязанность издать административный акт; смешанный текст классифицируется целиком."
    )
    left, middle, right = st.columns(3)
    left.metric("Тип", case["appeal_type"], help=case["classification_reasoning"])
    middle.metric("Уверенность", f"{case['confidence']:.0%}")
    right.metric("Риск эскалации", f"{case['escalation_risk']:.0%}", help=ENFORCEMENT_HELP)
    st.write("**Маршрут:** " + " → ".join(case["routing_targets"]))
    st.metric(
        "Срок / осталось",
        f"{case['deadline']} · {case['working_days_remaining']} раб. дн.",
        help=legal_tooltip(case["appeal_type"]),
    )
    if case["deemed_refusal"]:
        st.error("Считается отказом — АППК ст. 91(2)")
    if case["needs_human_review"]:
        st.warning("Требуется проверка оператором")
    st.write("**Факторы риска**")
    if case["risk_factors"]:
        st.markdown("\n".join(f"- {factor}" for factor in case["risk_factors"]))
    else:
        st.caption("Повышающие факторы не выявлены.")
    st.markdown(f'<div class="draft-banner">{DRAFT_BANNER}</div>', unsafe_allow_html=True)
    st.text_area("Проект ответа", case["draft_response"], height=220, key=f"draft-{case['id']}")

    cluster = next(item for item in clusters if item["cluster_id"] == case["cluster_id"])
    if st.button("Отметить кластер решённым", key=f"resolve-{case['cluster_id']}"):
        cluster["resolved"] = True
        cluster["notification_messages"] = generate_cluster_notifications(cluster, st.session_state.cases)
    if cluster.get("resolved"):
        messages = cluster.get("notification_messages") or generate_cluster_notifications(cluster, st.session_state.cases)
        st.write(f"**Уведомления заявителям: {len(messages)}**")
        copy_text = "\n\n".join(item["message"] for item in messages)
        st.code(copy_text, language=None)
        st.caption("Используйте значок копирования в блоке выше — сообщения не отправляются системой.")
        csv_bytes = pd.DataFrame(messages).to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать CSV уведомлений", csv_bytes, f"{cluster['cluster_id']}-notifications.csv", "text/csv")


def queue_tab(cases: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    st.subheader("Приём и очередь")
    paste = st.text_area("Вставьте обращения — по одному на строку", height=100, placeholder="Только синтетические данные; не вставляйте реальные персональные сведения.")
    upload = st.file_uploader("Или загрузите .txt, .csv, .json", type=["txt", "csv", "json"])
    load_col, process_col, _ = st.columns([1.45, 1.3, 3])
    demo_ready = DEMO_RESULTS.exists()
    if load_col.button(
        "Загрузить демо-набор (250 обращений)",
        type="primary", width="stretch", disabled=not demo_ready,
        help=None if demo_ready else "Замороженные модельные результаты — Скоро.",
    ):
        try:
            with st.spinner("Загрузка безопасного офлайн-набора…"):
                loaded_cases, loaded_clusters, warnings = _load_frozen_demo()
            st.session_state.cases, st.session_state.clusters = loaded_cases, loaded_clusters
            st.session_state.warnings = warnings
            st.rerun()
        except Exception:
            st.warning("Не удалось загрузить демо-набор. Проверьте целостность JSON-файлов.")
    if not demo_ready:
        st.caption("Замороженные модельные результаты — Скоро. Живой ввод уже работает в резервном режиме.")
    if process_col.button("Обработать введённые обращения", width="stretch"):
        try:
            texts = [line.strip() for line in paste.splitlines() if line.strip()] + _parse_upload(upload)
            if not texts:
                st.warning("Добавьте хотя бы одно обращение.")
            else:
                with st.spinner("Классификация и маршрутизация…"):
                    loaded_cases, loaded_clusters, warnings = process_records(_live_records(texts))
                st.session_state.cases, st.session_state.clusters = loaded_cases, loaded_clusters
                st.session_state.warnings = warnings
                st.rerun()
        except Exception:
            st.warning("Файл не распознан или обработка недоступна. Остальная очередь не затронута.")

    if not cases:
        st.info("Очередь пуста. Загрузите демо-набор или добавьте синтетические обращения.")
        return
    if st.session_state.get("warnings"):
        st.warning(FALLBACK_NOTICE)

    appeal_types = st.multiselect("Тип", sorted({case["appeal_type"] for case in cases}))
    topics = st.multiselect("Тема", sorted({case["topic"] for case in cases}))
    risk_bands = st.multiselect("Риск", ["green", "amber", "red"], format_func=lambda value: RISK_LABELS[value])
    f1, f2, f3 = st.columns(3)
    overdue_only = f1.checkbox("Только просроченные")
    cluster_only = f2.checkbox("Только кластеры (2+)")
    review_only = f3.checkbox("Требуют проверки")
    cluster_sizes = {cluster["cluster_id"]: cluster["member_count"] for cluster in clusters}
    filtered = [case for case in cases if
        (not appeal_types or case["appeal_type"] in appeal_types)
        and (not topics or case["topic"] in topics)
        and (not risk_bands or case["risk_band"] in risk_bands)
        and (not overdue_only or case["deemed_refusal"])
        and (not cluster_only or cluster_sizes[case["cluster_id"]] >= 2)
        and (not review_only or case["needs_human_review"])]

    rows = [{
        "ID": case["id"], "Тип": case["appeal_type"],
        "Тема": ROUTING.get(case["topic"], ROUTING["НЕ ОПРЕДЕЛЕНО"])["display_name"],
        "Ответственный": case["statutory_clock_holder"], "Срок": case["deadline"],
        "Осталось (раб. дн.)": case["working_days_remaining"], "Риск": RISK_LABELS[case["risk_band"]],
        "Кластер": f"{case['cluster_id']} · {cluster_sizes[case['cluster_id']]} шт.",
    } for case in filtered]
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("По выбранным фильтрам обращений нет.")
        return
    styled = frame.style.map(_deadline_color, subset=["Осталось (раб. дн.)"])
    event = st.dataframe(
        styled, hide_index=True, width="stretch", height=430,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Срок": st.column_config.TextColumn("Срок", help="АППК ст. 76(1) — 15 рабочих дней; ст. 99 — 20 рабочих дней для жалобы."),
            "Осталось (раб. дн.)": st.column_config.NumberColumn("Осталось (раб. дн.)", help="Счёт начинается на следующий день после регистрации — АППК ст. 76(2)."),
        },
    )
    selected_rows = list(event.selection.rows) if hasattr(event, "selection") else []
    if selected_rows:
        _case_detail(filtered[selected_rows[0]], clusters)


def map_tab(cases: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    st.subheader("Карта повторяющихся проблем")
    if not cases:
        st.info("Загрузите обращения на вкладке «Очередь», чтобы построить карту.")
        return
    view = st.radio("Режим отображения", ["Показать все обращения", "Показать кластеры"], horizontal=True)
    fmap = folium.Map(
        location=[53.284, 69.395], zoom_start=12,
        tiles=None, min_zoom=11, max_zoom=15,
        min_lat=53.19, max_lat=53.36, min_lon=69.28, max_lon=69.57,
        max_bounds=True, control_scale=True,
    )
    _embed_leaflet_assets(fmap)
    folium.TileLayer(
        tiles="/app/static/tiles/{z}/{x}/{y}.png",
        attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
        name="OpenStreetMap · офлайн", min_zoom=11, max_zoom=15,
        max_native_zoom=15, no_wrap=True, overlay=False, control=False,
    ).add_to(fmap)
    folium.Polygon(
        locations=[[53.300, 69.345], [53.312, 69.355], [53.316, 69.378], [53.306, 69.384], [53.297, 69.366]],
        color="#6099b5", fill=True, fill_color="#b9dcea", fill_opacity=.8,
        tooltip="озеро Копа (схематично)",
    ).add_to(fmap)
    folium.PolyLine(
        [[53.258, 69.352], [53.286, 69.359], [53.322, 69.348]],
        color="#6099b5", weight=3, tooltip="р. Шағалалы",
    ).add_to(fmap)
    folium.PolyLine(
        [[53.258, 69.432], [53.286, 69.420], [53.322, 69.439]],
        color="#6099b5", weight=3, tooltip="р. Кылшакты",
    ).add_to(fmap)
    city_layer = folium.FeatureGroup(name="Кокшетау", show=True)
    rural_layer = folium.FeatureGroup(name="Красный Яр / Кызыл-Жулдыз", show=True)
    station_layer = folium.FeatureGroup(name="Станционный", show=True)
    layer_for = {"Красный Яр": rural_layer, "Кызыл-Жулдыз": rural_layer, "Станционный": station_layer}
    rendered = 0
    by_id = {case["id"]: case for case in cases}
    if view == "Показать все обращения":
        for case in cases:
            location = case.get("location")
            if not location:
                continue
            popup = f"{html.escape(case['id'])} · {html.escape(case['appeal_type'])} · {case['working_days_remaining']} раб. дн."
            folium.CircleMarker(
                [location["lat"], location["lon"]], radius=4,
                color=RISK_COLORS[case["risk_band"]], fill=True, fill_opacity=.72,
                popup=folium.Popup(popup, max_width=360),
            ).add_to(layer_for.get(location["settlement"], city_layer))
            rendered += 1
    else:
        for cluster in clusters:
            member_cases = [by_id[member_id] for member_id in cluster["member_ids"]]
            located = [case for case in member_cases if case.get("location")]
            if not located:
                continue
            representative = located[0]
            location = representative["location"]
            highest_risk = max(member_cases, key=lambda case: case["escalation_risk"])
            oldest_days = max(1, (date.today() - datetime.fromisoformat(cluster["oldest_received_at"]).date()).days + 1)
            remaining = working_days_between(date.today(), date.fromisoformat(cluster["earliest_deadline"]))
            popup = (
                f"{cluster['member_count']} обращений · старейшее {oldest_days} дней · "
                f"срок истекает через {remaining} рабочих дней · {html.escape(representative['statutory_clock_holder'])}"
            )
            folium.CircleMarker(
                [location["lat"], location["lon"]],
                radius=max(6, min(28, 5 + math.sqrt(cluster["member_count"]) * 4)),
                color=RISK_COLORS[highest_risk["risk_band"]], fill=True, fill_opacity=.78,
                weight=2, popup=folium.Popup(popup, max_width=460), tooltip=f"{cluster['cluster_id']} · {cluster['member_count']} обращений",
            ).add_to(layer_for.get(location["settlement"], city_layer))
            rendered += 1
    city_layer.add_to(fmap)
    rural_layer.add_to(fmap)
    station_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st.caption(f"На карте: {rendered} отметок. Переключение схлопывает повторные обращения по одному объекту.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="(?s).*folium_static is deprecated.*", category=DeprecationWarning)
        folium_static(fmap, width=900, height=600)
    st.markdown('<div class="map-caption">Внутренний инструмент диспетчеризации. Не предназначен для публикации.</div>', unsafe_allow_html=True)


def deadlines_tab(cases: list[dict[str, Any]]) -> None:
    st.subheader("Контроль процессуальных сроков")
    if not cases:
        st.info("Загрузите обращения на вкладке «Очередь», чтобы увидеть контроль сроков.")
        return
    ordered = sorted(cases, key=lambda case: (not case["deemed_refusal"], case["working_days_remaining"]))
    overdue = sum(case["deemed_refusal"] for case in ordered)
    if overdue:
        st.error(f"Считается отказом — АППК ст. 91(2): {overdue} дел")
    for holder in sorted({case["statutory_clock_holder"] for case in ordered}):
        department_cases = [case for case in ordered if case["statutory_clock_holder"] == holder]
        with st.expander(f"{holder} · {len(department_cases)} дел", expanded=holder == "Отдел ЖКХ, ПТ и АД"):
            board = pd.DataFrame([{
                "ID": case["id"], "Тип": case["appeal_type"], "Срок": case["deadline"],
                "Осталось": case["working_days_remaining"], "Основание": case["deadline_basis"],
                "Статус": "Считается отказом — ст. 91(2)" if case["deemed_refusal"] else "В производстве",
            } for case in department_cases])
            st.dataframe(board.style.map(_deadline_color, subset=["Осталось"]), hide_index=True, width="stretch")

    st.subheader("Продление срока")
    selected_id = st.selectbox("Дело", [case["id"] for case in ordered], format_func=lambda value: next(
        f"{case['id']} · {case['appeal_type']} · {case['statutory_clock_holder']}" for case in ordered if case["id"] == value
    ))
    selected = next(case for case in ordered if case["id"] == selected_id)
    is_complaint = selected["appeal_type"] == "жалоба"
    if is_complaint:
        st.warning("Продление недоступно: срок рассмотрения жалобы не продлевается — АППК ст. 99.")
    facts_required = st.checkbox(
        "Установление фактических обстоятельств требует дополнительного времени",
        disabled=is_complaint,
        help="АППК ст. 76(3); необоснованное продление влечёт дисциплинарную ответственность по ст. 76(4).",
    )
    authoriser = st.selectbox(
        "Уполномоченный авторизатор",
        ["— выберите —", "руководитель", "заместитель", "руководитель аппарата"],
        disabled=is_complaint or not facts_required,
    )
    if not is_complaint and facts_required and authoriser != "— выберите —":
        new_date = extension_deadline(date.today())
        notice_date = notification_deadline(date.today())
        c1, c2 = st.columns(2)
        c1.metric("Новый предельный срок", new_date.isoformat(), help="Не более двух месяцев от даты решения о продлении — АППК ст. 76(3).")
        c2.metric("Уведомить заявителя до", notice_date.isoformat(), help="Отдельный срок: 3 рабочих дня — АППК ст. 76(3).")


def summary_tab(cases: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    st.subheader("Сводка по диспетчеризации")
    if not cases:
        st.info("Загрузите обращения на вкладке «Очередь», чтобы увидеть сводные показатели.")
        return
    duplicate_clusters = [cluster for cluster in clusters if cluster["member_count"] >= 2]
    collapsed = sum(cluster["member_count"] - 1 for cluster in duplicate_clusters)
    savings = sum(case["misroute_cost_avoided"] for case in cases)
    in_time = sum(not case["deemed_refusal"] for case in cases) / len(cases)
    high_risk = sum(case["risk_band"] == "red" for case in cases)
    unverified = sum(case["topic"] == "НЕ ОПРЕДЕЛЕНО" for case in cases)
    st.markdown(
        f'<div class="metric-hero"><div class="value">{savings:,}</div><div class="label">Сэкономлено рабочих дней за счёт корректной маршрутизации</div></div>'.replace(",", " "),
        unsafe_allow_html=True,
    )
    st.caption("Метод: до 3 рабочих дней по АППК ст. 65(1) × каждое дело с определённой компетенцией; это верхняя оценка предотвращённого перенаправления.")
    cols = st.columns(5)
    cols[0].metric("Обращений обработано", len(cases))
    cols[1].metric("Кластеров / схлопнуто", f"{len(duplicate_clusters)} / {collapsed}")
    cols[2].metric("В срок", f"{in_time:.0%}")
    cols[3].metric("Высокий риск", high_risk, help=ENFORCEMENT_HELP)
    cols[4].metric("Уточнить компетенцию", unverified)
    try:
        score = json.loads(DEMO_SCORE.read_text(encoding="utf-8"))
        result_digest = hashlib.sha256(DEMO_RESULTS.read_bytes()).hexdigest()
        accuracy = (
            score.get("accuracy", {}).get("appeal_type")
            if score.get("results_sha256") == result_digest
            else None
        )
    except (OSError, ValueError, TypeError):
        accuracy = None
    if isinstance(accuracy, (int, float)):
        st.metric("Точность типа обращения на демо-наборе", f"{accuracy:.1%}")
    st.subheader("Язык обращений")
    names = {"ru": "Русский", "kk": "Қазақша", "mixed": "Смешанные", "latin": "Латиница"}
    counts = pd.Series([case["language_detected"] for case in cases]).value_counts()
    chart = pd.DataFrame({
        "Язык": [names[key] for key in ("ru", "kk", "mixed", "latin")],
        "Количество": [int(counts.get(key, 0)) for key in ("ru", "kk", "mixed", "latin")],
        "Цвет": ["#8aa1b1", "#8aa1b1", "#087f78", "#8aa1b1"],
    })
    st.bar_chart(chart, x="Язык", y="Количество", color="Цвет", horizontal=True)
    st.caption("Смешанные казахско-русские обращения выделены: они классифицируются без предварительного перевода.")


st.title("JAUAP · диспетчеризация обращений")
st.markdown('<div class="demo-banner">Демонстрационная версия. Данные синтетические.</div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("Режим")
    forced_offline = os.environ.get("JAUAP_OFFLINE") == "1"
    mode = st.radio("Источник обработки", ["Демо · офлайн", "Живой ввод · Anthropic"], index=0)
    st.session_state["offline_mode"] = forced_offline or mode.startswith("Демо")
    st.caption("Ввод API-ключа в интерфейсе — Скоро.")
    if forced_offline:
        st.caption("JAUAP_OFFLINE=1: демо-набор читает только замороженные результаты; живой ввод проверяет ключ окружения.")
    elif mode.startswith("Живой"):
        st.caption("Пока используется ANTHROPIC_API_KEY из окружения; без него — резервные правила.")
    st.divider()
    st.caption("КАТО 111010000 · Кокшетау · внутренний операторский контур")

st.session_state.setdefault("cases", [])
st.session_state.setdefault("clusters", [])
st.session_state.setdefault("warnings", [])
tabs = st.tabs(["Очередь", "Карта", "Сроки", "Сводка"])
with tabs[0]:
    queue_tab(st.session_state.cases, st.session_state.clusters)
with tabs[1]:
    map_tab(st.session_state.cases, st.session_state.clusters)
with tabs[2]:
    deadlines_tab(st.session_state.cases)
with tabs[3]:
    summary_tab(st.session_state.cases, st.session_state.clusters)
