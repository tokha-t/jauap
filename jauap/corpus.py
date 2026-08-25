"""Deterministically generate the 250 synthetic appeals used by the demo.

This is a one-time generation script. Runtime code reads data/demo_corpus.json.
Every Kazakh and mixed-language record requires Tokha's review before a demo.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .deadline_engine import is_working_day


SEED = 221_241
AS_OF = date(2026, 8, 26)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_corpus.json"
CHANNELS = ["eOtinish", "109", "WhatsApp", "Telegram", "приём граждан"]
NAMES = [
    "Айгүл Серікқызы", "Шыңғыс Омаров", "Данияр Ахметов", "Мадина Қасымова",
    "Ольга Петрова", "Сергей Иванов", "Әлия Нұрланқызы", "Николай Смирнов",
    "Аружан Бекова", "Тимур Садыков", "Елена Ким", "Руслан Жақыпов",
]
CITY_ADDRESSES = [
    ("Шокана Уалиханова", 175), ("Мухтара Ауэзова", 42),
    ("Бауыржана Момышулы", 21), ("Еркина Ауельбекова", 88),
    ("Сакена Сейфуллина", 31), ("Каныша Сатпаева", 54),
    ("Акана серэ", 19), ("Гоголя", 67), ("Жибек жолы", 12),
    ("Маншук Маметовой", 24), ("Пушкина", 40), ("Шевченко", 73),
]

TOPIC_QUOTAS = {
    "water_supply": 33,
    "heating": 17,
    "sewerage": 15,
    "electricity": 10,
    "waste_removal": 35,
    "snow_cleaning": 15,
    "road_condition": 37,
    "street_lighting": 25,
    "landscaping": 25,
    "public_transport": 12,
    "construction": 6,
    "land": 6,
    "social": 7,
    "education": 6,
}

LANGUAGE_QUOTAS = {"ru": 100, "kk": 63, "mixed": 75, "latin": 12}

RU = {
    "water_supply": "На улице {street}, дом {building}, опять нет воды, сколько можно ждать???",
    "heating": "В доме по адресу {street} {building} батареи холодные, разберитесь пожалуйста",
    "sewerage": "Возле дома {building} по {street} течет канализация и ужасный запах!!!",
    "electricity": "На {street} {building} со вчерашнего вечера нет света, кто вообще отвечает?",
    "waste_removal": "Во дворе дома {building} по {street} уже неделю не вывозят мусор",
    "snow_cleaning": "{street} дом {building} не чистят, пройти невозможно, один лёд",
    "road_condition": "Огромная яма у дома {street} {building}, машины объезжают по тротуару",
    "street_lighting": "У {street} {building} не горят фонари, вечером совсем темно",
    "landscaping": "Во дворе {street} {building} сухое дерево висит над детской площадкой",
    "public_transport": "Автобус у остановки {street} {building} постоянно пропускает рейсы",
    "construction": "Когда начнётся обещанный капитальный ремонт возле {street} {building}?",
    "land": "Прошу объяснить границы земельного участка по адресу {street} {building}",
    "social": "Не пришла социальная выплата, живу по адресу {street} {building}, помогите",
    "education": "В школе рядом с {street} {building} протекает крыша, дети сидят в сырости",
}

KK = {
    "water_supply": "{street} көшесі {building}-үйде су жоқ, қашан қалпына келтіресіздер???",
    "heating": "{street} көшесі {building}-үйдің батареялары салқын, шара қолданыңыздар",
    "sewerage": "{street} көшесі {building}-үйдің жанында кәріз суы ағып жатыр!!!",
    "electricity": "{street} көшесі {building}-үйде кешеден бері жарық жоқ",
    "waste_removal": "{street} көшесі {building}-үйдің ауласынан қоқыс бір апта шығарылмады",
    "snow_cleaning": "{street} көшесі {building}-үй маңындағы қар тазаланбаған, жүру қиын",
    "road_condition": "{street} көшесі {building}-үйдің алдында үлкен шұңқыр бар",
    "street_lighting": "{street} көшесі {building}-үй маңындағы шамдар жанбайды, кешке қараңғы",
    "landscaping": "{street} көшесі {building}-үй ауласындағы қураған ағаш қауіпті болып тұр",
    "public_transport": "{street} көшесіндегі аялдамаға автобус уақытында келмейді",
    "construction": "{street} көшесі {building}-үй маңындағы күрделі жөндеу қашан басталады?",
    "land": "{street} көшесі {building}-үй жер телімінің шекарасын түсіндіруді сұраймын",
    "social": "{street} көшесі {building}-үйде тұрамын, әлеуметтік төлем түспеді",
    "education": "{street} көшесі {building}-үй маңындағы мектептің шатыры ағып тұр",
}

MIXED = {
    "water_supply": "{street} көшесі {building}-үйде опять вода жоқ, балалармен как жить???",
    "heating": "{street} {building} үйде отопление мүлдем слабое, примите меры",
    "sewerage": "{street} көшесі {building} возле дома канализация ағып жатыр, иіс ужасный",
    "electricity": "{street} {building} жарық жоқ со вчера, продукты портятся уже",
    "waste_removal": "{street} {building} ауласында мусор уже неделю жатыр, уберите пожалуйста",
    "snow_cleaning": "{street} көшесі {building} қар тазаланбаған, невозможно пройти",
    "road_condition": "{street} {building} алдында огромная яма, көлік өте алмайды",
    "street_lighting": "{street} {building} маңында фонари жанбайды, вечером страшно",
    "landscaping": "{street} {building} ауласында сухое дерево тұр, балаларға опасно",
    "public_transport": "{street} аялдамасында автобус опять жоқ, адамдар долго ждут",
    "construction": "{street} {building} маңындағы капремонт қашан болады, дайте ответ",
    "land": "{street} {building} жер учаскесінің границы түсініксіз, прошу разъяснить",
    "social": "Соцвыплата түспеді, {street} {building} тұрамын, помогите разобраться",
    "education": "{street} {building} жанындағы мектепте крыша ағып тұр, дети мерзнут",
}

LATIN = {
    "water_supply": "{street} {building} uide su joq, qashan beresizder???",
    "heating": "{street} {building} uide jylu joq, batareya salqyn",
    "sewerage": "{street} {building} janinda kariz suyi agyp jatyr",
    "electricity": "{street} {building} uide jaryq joq kesheden beri",
    "waste_removal": "{street} {building} aulasynda qoqys bir apta jatyr",
    "snow_cleaning": "{street} {building} qar tazalanbagan, juru qiyn",
    "road_condition": "{street} {building} aldinda ulken shungqyr bar",
    "street_lighting": "{street} {building} fonarlar janbaidy, keshe qarangy",
    "landscaping": "{street} {building} aulasynda qurgan agash qauipti",
    "public_transport": "{street} ayaldamasyna avtobus uaqytynda kelmeidi",
    "construction": "{street} {building} kapitaldy jondeu qashan bastalady?",
    "land": "{street} {building} jer shekarasyn tusindirinizder",
    "social": "{street} {building} turamyn, aleumettik tolem tuspedi",
    "education": "{street} {building} mektep shatyry agyp tur",
}


def _working_dates(count: int = 25) -> list[date]:
    result: list[date] = []
    current = AS_OF
    while len(result) < count:
        if is_working_day(current):
            result.append(current)
        current -= timedelta(days=1)
    return result


def _received_at(rng: random.Random, index: int) -> str:
    days = _working_dates()
    chosen = days[index % len(days)]
    stamp = datetime.combine(chosen, time(rng.randint(8, 19), rng.choice([0, 15, 30, 45])))
    return stamp.isoformat()


def _record(
    number: int,
    text: str,
    language: str,
    topic: str,
    appeal_type: str = "сообщение",
    hard_case: str | None = None,
    settlement: str = "Кокшетау",
    routing_targets: list[str] | None = None,
    confidence: float = 0.94,
) -> dict:
    rng = random.Random(SEED + number)
    return {
        "id": f"AP-{number:04d}",
        "raw_text": text,
        "received_at": _received_at(rng, number - 1),
        "channel": CHANNELS[(number - 1) % len(CHANNELS)],
        "applicant_name": NAMES[(number - 1) % len(NAMES)],
        "language_detected": language,
        "synthetic": True,
        "hard_case": hard_case,
        "expected": {
            "appeal_type": appeal_type,
            "topic": topic,
            "settlement": settlement,
            "routing_targets": routing_targets,
            "confidence": confidence,
        },
    }


def _hard_cases() -> list[dict]:
    pipe_phrasings = [
        "ул. Абая 14 — во дворе прорвало трубу, вода льётся уже третий день",
        "Абая көшесі 14-үй алдында труба жарылып су ағып жатыр",
        "Абая, д.14 опять порыв трубы во дворе, никто не приезжает",
        "абая 14 су құбыры сломан, весь двор в воде",
        "во дворе 14 дома по абая течет вода из сломанной трубы",
    ]
    records = []
    languages = ["ru", "mixed", "ru", "mixed", "ru", "kk", "mixed", "ru", "kk", "mixed",
                 "ru", "kk", "mixed", "ru", "kk", "mixed", "ru", "kk", "mixed", "ru"]
    for index in range(20):
        text = pipe_phrasings[index % len(pipe_phrasings)]
        if index:
            text += ["!!!", " пожалуйста помогите", " уже невозможно", " балалар жүре алмайды"][index % 4]
        records.append(_record(
            index + 1, text, languages[index], "water_supply",
            hard_case="broken_pipe_cluster_20" if index else "code_switched_naive_misroute+broken_pipe_cluster_20",
        ))

    records.extend([
        _record(21, "Красный Яр, ул. Достык 12 — вечером не горят фонари", "ru", "street_lighting",
                hard_case="rural_sub_akimat_1", settlement="Красный Яр",
                routing_targets=["Аппарат акима Красноярского сельского округа"]),
        _record(22, "Красный Яр Абая көшесі 7-үйде су жоқ, тезірек көмектесіңіздер", "kk", "water_supply",
                hard_case="rural_sub_akimat_2", settlement="Красный Яр",
                routing_targets=["Аппарат акима Красноярского сельского округа"]),
        _record(23, "Кызыл-Жулдыз ауылында дорога разбита және мусор шығарылмайды", "mixed", "road_condition",
                hard_case="rural_sub_akimat_3", settlement="Кызыл-Жулдыз",
                routing_targets=["Аппарат акима Красноярского сельского округа"]),
        _record(24, "На ул. Пушкина 40 огромная яма, а рядом незаконно поставили торговый павильон. Прошу решить оба вопроса.",
                "ru", "road_condition", hard_case="multi_request_split_65_2",
                routing_targets=["Отдел ЖКХ, ПТ и АД", "Отдел земельных отношений, архитектуры и градостроительства"]),
        _record(25, "Отдел ЖКХ месяц не отвечает на мое заявление о воде по ул. Гоголя 67. Требую признать бездействие и восстановить подачу.",
                "ru", "water_supply", appeal_type="жалоба", hard_case="complaint_20_days_no_extension"),
        _record(26, "Когда планируется ремонт улицы Мухтара Ауэзова? Прошу предоставить утвержденный график.",
                "ru", "road_condition", appeal_type="запрос", hard_case="information_request"),
        _record(27, "Во дворе ул. Шевченко 73 бегает стая бездомных собак, дети боятся выходить", "ru", "НЕ ОПРЕДЕЛЕНО",
                hard_case="unverified_competence"),
        _record(28, "Я уже писал про холодные батареи на Сатпаева 54, ответа нет. Прошу помочь или это уже отказ?",
                "ru", "heating", appeal_type="жалоба", hard_case="ambiguous_application_or_complaint", confidence=0.62),
    ])
    return records


def generate() -> list[dict]:
    rng = random.Random(SEED)
    records = _hard_cases()
    used_topics = Counter(record["expected"]["topic"] for record in records if record["expected"]["topic"] in TOPIC_QUOTAS)
    used_languages = Counter(record["language_detected"] for record in records)

    topic_pool = [topic for topic, quota in TOPIC_QUOTAS.items() for _ in range(quota - used_topics[topic])]
    language_pool = [lang for lang, quota in LANGUAGE_QUOTAS.items() for _ in range(quota - used_languages[lang])]
    rng.shuffle(topic_pool)
    rng.shuffle(language_pool)
    assert len(records) + len(topic_pool) == 250
    assert len(topic_pool) == len(language_pool)

    template_sets = {"ru": RU, "kk": KK, "mixed": MIXED, "latin": LATIN}
    for topic, language in zip(topic_pool, language_pool):
        number = len(records) + 1
        street, building = rng.choice(CITY_ADDRESSES)
        text = template_sets[language][topic].format(street=street, building=building)
        if number % 11 == 0:
            text = text.upper()
        elif number % 7 == 0:
            text += rng.choice(["!!!", " ну сколько можно", " жауап беріңіздер", " срочно пожалуйста"])
        records.append(_record(number, text, language, topic))

    assert len(records) == 250
    return records


def validate(records: list[dict]) -> None:
    assert len(records) == 250
    assert len({record["id"] for record in records}) == 250
    assert Counter(record["language_detected"] for record in records) == LANGUAGE_QUOTAS
    expected_topics = Counter(record["expected"]["topic"] for record in records)
    for topic, count in TOPIC_QUOTAS.items():
        assert expected_topics[topic] == count, (topic, expected_topics[topic], count)
    hard_cases = Counter(record["hard_case"] for record in records if record["hard_case"])
    assert sum("broken_pipe_cluster_20" in (record["hard_case"] or "") for record in records) == 20
    assert sum((record["hard_case"] or "").startswith("rural_sub_akimat") for record in records) == 3
    for required in ["multi_request_split_65_2", "complaint_20_days_no_extension", "information_request",
                     "unverified_competence", "ambiguous_application_or_complaint"]:
        assert hard_cases[required] == 1
    datetime_values = [datetime.fromisoformat(record["received_at"]) for record in records]
    assert min(value.date() for value in datetime_values) >= _working_dates()[-1]
    assert max(value.date() for value in datetime_values) <= AS_OF
    json.loads(json.dumps(records, ensure_ascii=False))


def main() -> None:
    records = generate()
    validate(records)
    DATA_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} synthetic appeals to {DATA_PATH}")
    print("Language distribution:", dict(Counter(record["language_detected"] for record in records)))
    print("MANDATORY: Tokha must review every kk, mixed, and latin appeal before the demo.")


if __name__ == "__main__":
    main()
