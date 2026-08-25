"""Official-response drafts and unsent cluster-resolution notifications."""

from __future__ import annotations

from typing import Any

from .llm import complete


DRAFT_BANNER = "ПРОЕКТ — требует проверки и подписи должностного лица"

DRAFT_SYSTEM = """Draft an official administrative response for an internal Kazakhstan akimat console.
Match the appeal language: Russian to Russian, Kazakh to Kazakh, and code-switched or Latin-transliterated Kazakh to Kazakh under Закон «О языках» ст. 9.
Use formal administrative register. Cite the supplied deadline basis and name the supplied responsible entity.
Include the literal placeholder [СРОК ИСПОЛНЕНИЯ] where a concrete date must be entered by a human.
Never fabricate a commitment, result, inspection, repair date, or promise. State procedure only.
Return only the draft body; do not add a signature."""


def _target_language(language: str) -> str:
    return "ru" if language == "ru" else "kk"


def _fallback_draft(case: dict[str, Any]) -> str:
    owner = case["statutory_clock_holder"]
    basis = case["deadline_basis"]
    if _target_language(case["language_detected"]) == "ru":
        return (
            "Уважаемый заявитель!\n\n"
            f"Ваше обращение зарегистрировано и направлено в компетенцию: {owner}. "
            f"Срок рассмотрения определяется {basis}. "
            "Информация о результатах рассмотрения будет оформлена уполномоченным должностным лицом "
            "в установленном порядке. Контрольная дата: [СРОК ИСПОЛНЕНИЯ].\n\n"
            "Настоящий проект описывает порядок рассмотрения и не содержит обещания конкретного результата "
            "или даты выполнения работ."
        )
    return (
        "Құрметті өтініш беруші!\n\n"
        f"Сіздің өтінішіңіз тіркеліп, мына құзыретті органға жолданды: {owner}. "
        f"Қарау мерзімі {basis} негізінде айқындалады. "
        "Қарау нәтижесі туралы ақпаратты уәкілетті лауазымды тұлға белгіленген тәртіппен рәсімдейді. "
        "Бақылау күні: [СРОК ИСПОЛНЕНИЯ].\n\n"
        "Бұл жоба қарау тәртібін ғана сипаттайды және нақты нәтиже немесе жөндеу мерзімі жөнінде уәде бермейді."
    )


def draft_response(case: dict[str, Any], *, frozen_demo: bool = False) -> str:
    if frozen_demo:
        return _fallback_draft(case)
    user = (
        f"Language: {case['language_detected']}\n"
        f"Appeal: {case['raw_text']}\n"
        f"Responsible entity: {case['statutory_clock_holder']}\n"
        f"Deadline basis: {case['deadline_basis']}"
    )
    try:
        result = complete(DRAFT_SYSTEM, user)
        if isinstance(result, str) and "[СРОК ИСПОЛНЕНИЯ]" in result:
            return result
    except Exception:
        pass
    return _fallback_draft(case)


def _notification(case: dict[str, Any]) -> str:
    if _target_language(case["language_detected"]) == "ru":
        return (
            f"По группе обращений, включающей Ваше обращение {case['id']}, оператор отметил вопрос как решённый. "
            "Сообщение сформировано автоматически и подлежит проверке перед отправкой."
        )
    return (
        f"Сіздің {case['id']} өтінішіңіз кіретін топ бойынша оператор мәселені шешілді деп белгіледі. "
        "Хабарлама автоматты түрде жасалды және жіберер алдында тексерілуге тиіс."
    )


def generate_cluster_notifications(cluster: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Generate, but never send, one message per distinct synthetic applicant."""
    member_ids = set(cluster["member_ids"])
    seen: set[str] = set()
    notifications: list[dict[str, str]] = []
    for case in cases:
        if case["id"] not in member_ids or case["applicant_name"] in seen:
            continue
        seen.add(case["applicant_name"])
        notifications.append({
            "applicant_name": case["applicant_name"],
            "appeal_id": case["id"],
            "language": _target_language(case["language_detected"]),
            "message": _notification(case),
        })
    return notifications
