"""Deterministically generate the 250 synthetic appeals used by the demo.

This is a one-time generation script. Runtime code reads data/demo_corpus.json.
Every Kazakh and mixed-language record requires Tokha's review before a demo.
"""

from __future__ import annotations

import json
import hashlib
import random
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .classify import _language
from .deadline_engine import is_working_day


SEED = 221_241
AS_OF = date(2026, 8, 26)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_corpus.json"
GROUND_TRUTH_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_ground_truth.json"
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
UNVERIFIED_TOPICS = {"housing_inspection", "electricity", "stray_animals"}

LANGUAGE_QUOTAS = {"ru": 100, "kk": 63, "mixed": 75, "latin": 12}
TYPE_QUOTAS = {
    "сообщение": 150,
    "заявление": 55,
    "жалоба": 25,
    "запрос": 12,
    "предложение": 5,
    "отклик": 3,
}

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

TYPE_WRAPPERS = {
    "ru": {
        "сообщение": "{base}. Сообщаю этот факт для регистрации.",
        "заявление": "{base}. Для моего дома нужно устранить эту проблему; прошу включить работу в исполнение.",
        "жалоба": "Ранее обращался по той же проблеме: {base}. Ответа нет; обжалую бездействие и добиваюсь восстановления нарушенного права.",
        "запрос": "{base}. Нужна информация: какой срок и график работ утверждены?",
        "предложение": "{base}. Предлагаю включить этот объект в план улучшений, чтобы ситуация не повторялась.",
        "отклик": "{base}. Поддерживаю принятое городом решение устранить эту проблему.",
    },
    "kk": {
        "сообщение": "{base}. Бұл жағдайды тіркеу үшін хабарлаймын.",
        "заявление": "{base}. Өз құқығымды іске асыру үшін мәселені жоюға жәрдем көрсетуді сұраймын.",
        "жалоба": "Осы мәселе бойынша бұрын өтініш бердім: {base}. Жауап болмады, әрекетсіздікке шағымданамын және бұзылған құқығымды қалпына келтіруді талап етемін.",
        "запрос": "{base}. Жұмыстың бекітілген мерзімі мен кестесі туралы ақпарат беруді сұраймын.",
        "предложение": "{base}. Мұндай жағдай қайталанбауы үшін нысанды жақсарту жоспарына енгізуді ұсынамын.",
        "отклик": "{base}. Осы мәселені шешу жөніндегі қала шешімін қолдаймын.",
    },
    "mixed": {
        "сообщение": "{base}. Уже бірнеше күн болды, фактіні тіркеуді прошу.",
        "заявление": "{base}. Өз құқығымды іске асыру үшін прошу включить устранение в работу.",
        "жалоба": "Раньше осы мәселе бойынша өтініш бердім: {base}. Ответа нет; обжалую әрекетсіздік и добиваюсь восстановления права.",
        "запрос": "{base}. Когда бекітілген мерзім мен график дайын болады, прошу сообщить.",
        "предложение": "{base}. Қайталанбауы үшін предлагаю нысанды жоспарға енгізуді, прошу рассмотреть.",
        "отклик": "{base}. Қаланың осы шешімін поддерживаю, пікірімді прошу учесть.",
    },
    "latin": {
        "сообщение": "{base}. Bul jagdaidy tirkeuge jiberip oturmyn.",
        "заявление": "{base}. Oz quqygymdy iske asyru ushin komek korsetuinizdi suraimyn.",
        "жалоба": "Buryngy otinishim jawapsyz qaldy: {base}. Areketsizdikke shagymdanamyn, quqygymdy qalpyna keltirudi suraimyn.",
        "запрос": "{base}. Bekitilgen merzim men jumys kestesi qashan dayin bolady?",
        "предложение": "{base}. Qaitalanbauy ushin nysandy jaqsartu josparyna engizudi usynamyn.",
        "отклик": "{base}. Qala sheshimin qoldaimyn, pikirimdi eskeruinizdi suraimyn.",
    },
}

TYPE_TEMPLATES = {
    language: {
        topic: dict(TYPE_WRAPPERS[language])
        for topic in topic_templates
    }
    for language, topic_templates in {
        "ru": RU,
        "kk": KK,
        "mixed": MIXED,
        "latin": LATIN,
    }.items()
}

QUERY_SUBJECTS = {
    "ru": {
        "water_supply": "водоснабжение", "heating": "отопление", "sewerage": "канализация",
        "electricity": "электроснабжение", "waste_removal": "вывоз мусора", "snow_cleaning": "уборка снега",
        "road_condition": "ремонт дороги", "street_lighting": "уличное освещение", "landscaping": "озеленение",
        "public_transport": "маршрут автобуса", "construction": "капитальный ремонт", "land": "земельный участок",
        "social": "социальная выплата", "education": "ремонт школы",
    },
    "kk": {
        "water_supply": "су құбыры", "heating": "жылу жүйесі", "sewerage": "кәріз жүйесі",
        "electricity": "электр қуаты", "waste_removal": "қоқыс шығару", "snow_cleaning": "қар тазалау",
        "road_condition": "жолды жөндеу", "street_lighting": "көшені жарықтандыру", "landscaping": "ағаштарды күту",
        "public_transport": "автобус бағыты", "construction": "күрделі жөндеу", "land": "жер телімі",
        "social": "әлеуметтік төлем", "education": "мектепті жөндеу",
    },
    "latin": {
        "water_supply": "su qubyry", "heating": "jylu juiesi", "sewerage": "kariz juiesi",
        "electricity": "elektr quaty", "waste_removal": "qoqys shygaru", "snow_cleaning": "qar tazalau",
        "road_condition": "jol jondeu", "street_lighting": "koshe jarygy", "landscaping": "agash kutimi",
        "public_transport": "avtobus bagyty", "construction": "kapitaldy jondeu", "land": "jer telimi",
        "social": "aleumettik tolem", "education": "mektep jondeu",
    },
}
QUERY_SUBJECTS["mixed"] = dict(QUERY_SUBJECTS["kk"])
QUERY_TEXT = {
    "ru": "По теме «{subject}» для объекта {street} {building} нужна информация: какой срок и график работ утверждены?",
    "kk": "{street} көшесі {building}-үйдегі «{subject}» тақырыбы бойынша бекітілген мерзім мен кесте туралы ақпарат беруді сұраймын.",
    "mixed": "{street} {building} нысанындағы «{subject}» бойынша нужна информация: бекітілген мерзім мен график когда будут готовы?",
    "latin": "{street} {building} nysanyndagy {subject} turaly aqparat kerek. Bekitilgen merzim men jumys kestesi qashan dayin bolady?",
}

ADVERSARIAL_TOPICS = [
    "water_supply",
    "heating",
    "sewerage",
    "waste_removal",
    "snow_cleaning",
    "road_condition",
    "street_lighting",
    "landscaping",
    "public_transport",
    "education",
]
ADVERSARIAL_LANGUAGES = ["ru", "kk", "mixed", "latin", "ru", "kk", "mixed", "latin", "ru", "kk"]
LANGUAGE_SUFFIXES = {
    "ru": ["!!!", " ну сколько можно", " срочно пожалуйста"],
    "kk": ["!!!", " жауап беріңіздер", " тезірек көмектесіңіздер"],
    "mixed": ["!!! уже невозможно", " жауап беріңіздер пожалуйста", " срочно көмектесіңіздер"],
    "latin": ["!!!", " jauap berinizder", " qashan sheshiledi"],
}
TEXT_VARIANTS = {
    "ru": [
        " Ситуация сохраняется на момент отправки.",
        " Проблема актуальна сегодня.",
        " Пишу от имени жильцов дома.",
        " Описываю состояние объекта без изменений.",
        " Это повторяется в нашем квартале.",
        " На месте всё остаётся по-прежнему.",
    ],
    "kk": [
        " Жағдай хабар жіберілген сәтте әлі өзгермеді.",
        " Мәселе бүгін де өзекті болып тұр.",
        " Үй тұрғындарының атынан жазып отырмын.",
        " Нысанның қазіргі күйін хабарлап отырмын.",
        " Бұл жағдай біздің аумақта қайталанып отыр.",
        " Оқиға орнында өзгеріс болған жоқ.",
    ],
    "mixed": [
        " Жағдай на момент отправки өзгерген жоқ.",
        " Мәселе сегодня да актуально болып тұр.",
        " Үй тұрғындары атынан пишу.",
        " Нысанның текущее состояние өзгермеді.",
        " Бұл жағдай в нашем квартале қайталанады.",
        " На месте әзірге өзгеріс жоқ.",
    ],
    "latin": [
        " Jagdai habar jiberilgen satte ali ozgergen joq.",
        " Masele bugin de ozekti, azirge sheshim joq.",
        " Ui turgyndary atynan jazyp oturmyn, ozgeris joq.",
        " Nysannyn qazirgi kuiin habarlap oturmyn, ozgeris joq.",
        " Bul jagdai bizdin aumaqta qaitalanyp tur, sheshim joq.",
        " Oqiga ornynda ali ozgeris joq.",
    ],
}
APPLICATION_VARIANTS = {
    "ru": [
        " Результат нужен для реализации права жильцов.",
        " Ожидаю административного решения по существу просьбы.",
        " Прошу рассмотреть это как заявление о содействии.",
        " Исполнение просьбы позволит нормально пользоваться жильём.",
        " Нужна помощь органа для восстановления услуги.",
        " Прошу принять решение и организовать исполнение.",
    ],
    "kk": [
        " Нәтиже тұрғындардың құқығын іске асыру үшін қажет.",
        " Өтініштің мәні бойынша әкімшілік шешім күтемін.",
        " Мұны жәрдем көрсету туралы өтініш ретінде қарауды сұраймын.",
        " Өтініш орындалса, тұрғын үйді қалыпты пайдалана аламыз.",
        " Қызметті қалпына келтіру үшін органның көмегі қажет.",
        " Шешім қабылдап, орындалуын ұйымдастыруды сұраймын.",
    ],
    "mixed": [
        " Нәтиже нужно для іске асыру құқығымызды.",
        " Өтініш мәні бойынша жду административное решение.",
        " Мұны заявление о содействии ретінде қарауды сұраймын.",
        " Просьба орындалса, тұрғын үйді қалыпты пайдалана аламыз.",
        " Қызметті қалпына келтіру үшін нужна помощь органа.",
        " Прошу принять решение және орындалуын ұйымдастыруды.",
    ],
    "latin": [
        " Natije quqyqty iske asyru ushin qajet, azirge sheshim joq.",
        " Otinish mani boiynsha akimshilik sheshim kerek, azirge joq.",
        " Muny jardem turaly otinish retinde qaraudy suraimyn, sheshim joq.",
        " Otinish oryndalsa, uidi qalypty paidalana alamyz, kedergi joq.",
        " Qyzmetti qalpyna keltiru ushin organnyn komegi kerek, sheshim joq.",
        " Sheshim qabyldap, oryndaluyin uiymdastyrudy suraimyn, azirge joq.",
    ],
}
CLUSTER_VARIANTS = {
    "ru": [
        " Жильцы заметили это утром.",
        " Вода продолжает течь к подъезду.",
        " Во дворе уже образовалась лужа.",
        " Аварийная бригада пока не приехала.",
    ],
    "mixed": [
        " Таңертең жильцы қайта көрді.",
        " Су әлі течёт к подъезду.",
        " Аулада уже большая лужа.",
        " Аварийная бригада әлі келген жоқ.",
    ],
}

KAZAKH_GLYPHS = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
KAZAKH_MORPHEMES = re.compile(
    r"\b(балалар|ауылында|және|жатыр|жоқ|бері|алдында|жанында|көмектесіңіздер|сұраймын|"
    r"хабарлаймын|ұсынамын|қолдаймын|шағымданамын|мәселе|жағдай|нысан|үй|аула)\b",
    re.IGNORECASE,
)
RUSSIAN_FUNCTION_WORDS = re.compile(
    r"\b(нет|уже|прошу|когда|дом|двор|опять|пожалуйста|требую|раньше|ответа|для|этот|этой|по|во|и|или|сегодня|пишу|текущее|нашем|месте|момент)\b",
    re.IGNORECASE,
)
RUSSIAN_LEXEMES = re.compile(
    r"\b(труба|вода|жильцы|сломан|ответ|мусор|фонари|автобус|крыша|батареи|проблема|ситуация)\b",
    re.IGNORECASE,
)
LATIN_KAZAKH_MARKERS = re.compile(
    r"\b(joq|qashan|uide|jatyr|tuspedi|janbaidy|jagdai|masele|turgyndary|nysan|ozgeris|qaitalanyp)\b",
    re.IGNORECASE,
)


def language_label_violations(text: str, label: str) -> list[str]:
    """Validate language labels in both directions without changing fallback logic."""
    has_kazakh = bool(KAZAKH_GLYPHS.search(text) or KAZAKH_MORPHEMES.search(text))
    has_russian_function = bool(RUSSIAN_FUNCTION_WORDS.search(text))
    has_russian = bool(has_russian_function or RUSSIAN_LEXEMES.search(text))
    has_latin_kazakh = bool(LATIN_KAZAKH_MARKERS.search(text))
    violations: list[str] = []
    if label == "ru" and has_kazakh:
        violations.append("ru label contains Kazakh glyphs or morphemes")
    elif label == "kk":
        if not has_kazakh:
            violations.append("kk label has no Kazakh glyph or morpheme")
        if has_russian_function:
            violations.append("kk label contains Russian function words")
    elif label == "mixed":
        if not has_kazakh:
            violations.append("mixed label has no Kazakh signal")
        if not has_russian:
            violations.append("mixed label has no Russian function word")
    elif label == "latin" and not has_latin_kazakh:
        violations.append("latin label has no Latin-script Kazakh marker")
    return violations


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
        "settlement": settlement,
        "_source_topic": topic,
        "_ground_truth": {
            "appeal_type": appeal_type,
            "topic": "НЕ ОПРЕДЕЛЕНО" if topic in UNVERIFIED_TOPICS else topic,
            "settlement": settlement,
            "routing_targets": routing_targets,
            "confidence": confidence,
        },
    }


def _hard_cases() -> list[dict]:
    pipe_phrasings = [
        ("ул. Абая 14 — во дворе прорвало трубу, вода льётся уже третий день", "ru"),
        ("Абая көшесі 14-үй алдында труба жарылып, су ағып жатыр", "mixed"),
        ("Абая, д.14 опять порыв трубы во дворе, никто не приезжает", "ru"),
        ("абая 14 су құбыры сломан, весь двор в воде", "mixed"),
        ("во дворе 14 дома по абая течет вода из сломанной трубы", "ru"),
    ]
    records = []
    for index in range(20):
        text, language = pipe_phrasings[index % len(pipe_phrasings)]
        text += CLUSTER_VARIANTS[language][index // len(pipe_phrasings)]
        record = _record(
            index + 1, text, language, "water_supply",
            hard_case="broken_pipe_cluster_20" if index else "code_switched_naive_misroute+broken_pipe_cluster_20",
        )
        cluster_day = AS_OF - timedelta(days=index % 9)
        record["received_at"] = datetime.combine(cluster_day, time(9 + index % 8, 15)).isoformat()
        records.append(record)

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


def _topic_pool(used_topics: Counter[str], rng: random.Random) -> list[str]:
    remaining = Counter({
        topic: quota - used_topics[topic]
        for topic, quota in TOPIC_QUOTAS.items()
    })
    adversarial: list[str] = []
    for topic in ADVERSARIAL_TOPICS:
        assert remaining[topic] >= 3
        adversarial.extend([topic] * 3)
        remaining[topic] -= 3
    ordinary = [topic for topic, count in remaining.items() for _ in range(count)]
    rng.shuffle(ordinary)
    return adversarial + ordinary


def _type_pool(used_types: Counter[str], rng: random.Random) -> list[str]:
    remaining = Counter({
        appeal_type: quota - used_types[appeal_type]
        for appeal_type, quota in TYPE_QUOTAS.items()
    })
    adversarial = [
        appeal_type
        for _ in ADVERSARIAL_TOPICS
        for appeal_type in ("сообщение", "заявление", "жалоба")
    ]
    for appeal_type in adversarial:
        remaining[appeal_type] -= 1
        assert remaining[appeal_type] >= 0
    ordinary = [appeal_type for appeal_type, count in remaining.items() for _ in range(count)]
    rng.shuffle(ordinary)
    return adversarial + ordinary


def _language_pool(
    type_pool: list[str], used_languages: Counter[str], rng: random.Random
) -> list[str]:
    capacity = Counter({
        language: quota - used_languages[language]
        for language, quota in LANGUAGE_QUOTAS.items()
    })
    assigned: list[str | None] = [None] * len(type_pool)

    for triple_index, language in enumerate(ADVERSARIAL_LANGUAGES):
        for offset in range(3):
            position = triple_index * 3 + offset
            assigned[position] = language
            capacity[language] -= 1

    required_languages = {
        "запрос": ["ru", "kk", "mixed", "latin"],
        "предложение": ["ru", "kk", "mixed", "latin"],
        # Three records cannot occupy four pools; prioritise the multilingual claim.
        "отклик": ["kk", "mixed", "latin"],
    }
    for appeal_type, languages in required_languages.items():
        positions = [
            index
            for index, assigned_language in enumerate(assigned)
            if assigned_language is None and type_pool[index] == appeal_type
        ]
        assert len(positions) >= len(languages)
        for position, language in zip(positions, languages):
            assigned[position] = language
            capacity[language] -= 1

    unassigned = [index for index, language in enumerate(assigned) if language is None]
    remaining_languages = [
        language
        for language, count in capacity.items()
        for _ in range(count)
    ]
    assert len(unassigned) == len(remaining_languages)
    rng.shuffle(remaining_languages)
    for position, language in zip(unassigned, remaining_languages):
        assigned[position] = language
    return [str(language) for language in assigned]


def generate() -> list[dict]:
    rng = random.Random(SEED)
    records = _hard_cases()
    used_topics = Counter(
        record["_source_topic"]
        for record in records
        if record["_source_topic"] in TOPIC_QUOTAS
    )
    used_languages = Counter(record["language_detected"] for record in records)
    used_types = Counter(record["_ground_truth"]["appeal_type"] for record in records)

    topic_pool = _topic_pool(used_topics, rng)
    type_pool = _type_pool(used_types, rng)
    language_pool = _language_pool(type_pool, used_languages, rng)
    assert len(records) + len(topic_pool) == 250
    assert len(topic_pool) == len(language_pool)
    assert len(topic_pool) == len(type_pool)
    used_texts = {record["raw_text"] for record in records}

    template_sets = {"ru": RU, "kk": KK, "mixed": MIXED, "latin": LATIN}
    for bulk_index, (topic, language, appeal_type) in enumerate(
        zip(topic_pool, language_pool, type_pool)
    ):
        number = len(records) + 1
        if bulk_index < len(ADVERSARIAL_TOPICS) * 3:
            triple_index = bulk_index // 3
            street, building = CITY_ADDRESSES[triple_index]
            hard_case = f"adversarial_type_triple_{triple_index + 1:02d}"
        else:
            street, building = rng.choice(CITY_ADDRESSES)
            hard_case = None
        if appeal_type == "запрос":
            text = QUERY_TEXT[language].format(
                subject=QUERY_SUBJECTS[language][topic],
                street=street,
                building=building,
            )
        else:
            text = template_sets[language][topic].format(street=street, building=building)
            text = TYPE_TEMPLATES[language][topic][appeal_type].format(base=text)
        if hard_case is None and number % 11 == 0:
            text = text.upper()
        variants = (
            APPLICATION_VARIANTS[language]
            if appeal_type == "заявление"
            else TEXT_VARIANTS[language]
        )
        start = number % len(variants)
        candidates = [
            text + variants[(start + offset) % len(variants)]
            for offset in range(len(variants))
        ]
        text = next((candidate for candidate in candidates if candidate not in used_texts), "")
        assert text, ("phrasing pool exhausted", number, language, topic, appeal_type)
        used_texts.add(text)
        records.append(
            _record(
                number,
                text,
                language,
                topic,
                appeal_type=appeal_type,
                hard_case=hard_case,
            )
        )

    assert len(records) == 250
    return records


def validate(records: list[dict]) -> None:
    assert len(records) == 250
    assert len({record["id"] for record in records}) == 250
    assert len({record["raw_text"] for record in records}) == 250
    assert Counter(record["language_detected"] for record in records) == LANGUAGE_QUOTAS
    for record in records:
        detected = _language(record["raw_text"])
        assert detected == record["language_detected"], (
            record["id"],
            record["language_detected"],
            detected,
            record["raw_text"],
        )
        violations = language_label_violations(
            record["raw_text"], record["language_detected"]
        )
        assert not violations, (
            record["id"],
            record["language_detected"],
            violations,
            record["raw_text"],
        )
    appeal_types = Counter(record["_ground_truth"]["appeal_type"] for record in records)
    assert appeal_types == TYPE_QUOTAS
    for language in LANGUAGE_QUOTAS:
        for topic in TOPIC_QUOTAS:
            assert set(TYPE_TEMPLATES[language][topic]) == set(TYPE_QUOTAS)
    for appeal_type, target in TYPE_QUOTAS.items():
        represented_languages = {
            record["language_detected"]
            for record in records
            if record["_ground_truth"]["appeal_type"] == appeal_type
        }
        assert len(represented_languages) == min(target, len(LANGUAGE_QUOTAS))
    expected_topics = Counter(record["_source_topic"] for record in records)
    for topic, count in TOPIC_QUOTAS.items():
        assert expected_topics[topic] == count, (topic, expected_topics[topic], count)
    hard_cases = Counter(record["hard_case"] for record in records if record["hard_case"])
    assert sum("broken_pipe_cluster_20" in (record["hard_case"] or "") for record in records) == 20
    assert sum((record["hard_case"] or "").startswith("rural_sub_akimat") for record in records) == 3
    adversarial_groups = {
        hard_case: [record for record in records if record["hard_case"] == hard_case]
        for hard_case in hard_cases
        if hard_case.startswith("adversarial_type_triple_")
    }
    assert len(adversarial_groups) == 10
    for group in adversarial_groups.values():
        assert len(group) == 3
        assert len({record["_ground_truth"]["topic"] for record in group}) == 1
        assert len({record["language_detected"] for record in group}) == 1
        assert {record["_ground_truth"]["appeal_type"] for record in group} == {
            "сообщение", "заявление", "жалоба"
        }
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
    appeal_type_distribution = Counter(
        record["_ground_truth"]["appeal_type"]
        for record in records
    )
    ground_truth = {
        record["id"]: record.pop("_ground_truth")
        for record in records
    }
    for record in records:
        record.pop("_source_topic")
    corpus_text = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    DATA_PATH.write_text(corpus_text, encoding="utf-8")
    GROUND_TRUTH_PATH.write_text(
        json.dumps(
            {
                "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
                "ground_truth": ground_truth,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} synthetic appeals to {DATA_PATH}")
    print(f"Wrote scoring truth to {GROUND_TRUTH_PATH}")
    print("Language distribution:", dict(Counter(record["language_detected"] for record in records)))
    print("Appeal type distribution:", dict(appeal_type_distribution))
    print("MANDATORY: Tokha must review every kk, mixed, and latin appeal before the demo.")


if __name__ == "__main__":
    main()
