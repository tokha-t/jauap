# JAUAP

JAUAP is a working demonstration of an internal citizen-request dispatch console for the Akimat of Kokshetau. It accepts synthetic requests in Russian, Kazakh, and mixed language, determines request type and responsible authority, calculates statutory deadlines, groups messages about the same object, explains escalation risk, and prepares draft replies.

**Live demo:** [jauap-demo.streamlit.app](https://jauap-demo.streamlit.app/)

> **All demonstration data is synthetic. Do not upload real personal information.**

## Run locally

Python 3.11 or newer is required.

```bash
pip install -r requirements.txt
JAUAP_OFFLINE=1 streamlit run app.py
```

The project’s primary provider is **Gemini** and remains selected until the project owner explicitly chooses another backend. The resumable `scripts/freeze_demo.py` script creates a frozen 250-record demo set through the real `classify_text()` path, saves a checkpoint after every request, and reuses its cache on reruns. The offline loader accepts results only when the provider, model, classification contract, and SHA-256 hash of the current corpus all match.

To classify your own synthetic text with the live model:

```bash
export JAUAP_PROVIDER="gemini"
export GOOGLE_API_KEY="..."
# export JAUAP_MODEL="gemini-3.5-flash-lite"  # optional override
streamlit run app.py
```

API-key entry in the interface is deferred and marked **Coming soon**. For now, live input reads the Gemini key only from the `GOOGLE_API_KEY` environment variable. Without a key, the app does not fail: it falls back to rules, assigns confidence `0.3`, and requires manual review.

If Gemini fails, every request remains in the queue. The app applies fallback classification with confidence `0.3` and requires manual review. Responses are cached in `data/.llm_cache/`; the cache key is a SHA-256 hash of the provider, model, system prompt, and user text. The internal boundary in `jauap/llm.py` is preserved so the backend can be replaced later through an explicit decision without changing the classifier or interface.

To freeze the full synthetic corpus once:

```bash
export JAUAP_PROVIDER="gemini"
export GOOGLE_API_KEY="..."
python scripts/freeze_demo.py --delay 4
python scripts/score_demo.py
```

`.env` is ignored by Git; `.env.example` contains empty values only. Keys are read exclusively from the environment.

## Data handling and evaluation

All demonstration requests are fully synthetic. The system has not processed any real citizen request. The actual `provider`, `model`, `generated_at`, and corpus SHA-256 hash are recorded in `data/demo_results.json` and validated before loading. The current Gemini `gemini-3.5-flash-lite` freeze reports **88.4%** request-type accuracy versus **60.0%** for the constant “message” baseline: a measured model contribution of **+28.4 percentage points**. Topic accuracy is **96.4%** and settlement accuracy is **100.0%**. The system flags **13.6%** of requests for review: request-type accuracy is **52.9%** on flagged cases and **94.0%** on the rest, a **41.0-point** gap. `scripts/score_demo.py` reproduces these figures against isolated ground truth; the interface displays them only when the result hash matches.

The [Google Gemini additional terms](https://ai.google.dev/gemini-api/terms) apply to unpaid Gemini usage. Google states that request and response content may be used to provide, improve, and develop Google products and technologies, and that human reviewers may read, annotate, and process API inputs and outputs. The terms expressly state: **“Do not submit sensitive, confidential, or personal information to the Unpaid Services.”**

Free providers are therefore suitable only for synthetic demonstrations. Before the first real request enters the system—before a weeks 5–8 pilot—the backend must move to a paid plan with appropriate data-processing terms or to local inference inside the Akimat’s perimeter. Under Kazakhstan’s personal-data law, the Akimat is the controller of real requests; unpaid-plan terms are incompatible with that role.

## Demo walkthrough

After creating and scoring honest frozen results:

1. On **Queue**, load 250 requests and open a mixed Kazakh–Russian request.
2. On **Map**, switch from individual markers to clusters. Twenty descriptions of a burst pipe in the courtyard of building 14 on Abay Street form exactly one `CL-001` cluster; the actual marker count comes from the current frozen results.
3. On **Deadlines**, select complaint `AP-0025`: the deadline is 20 working days and extension is blocked under APPC Article 99.
4. On **Summary**, show the upper-bound estimate of prevented routing delay and the share of mixed-language requests.

The map is an internal operator view only. It provides no publication, sharing, public export, or embedding workflow.

## Deadline rules

The rules are implemented in `jauap/deadline_engine.py` from the specification table:

- application — 15 working days, APPC Article 76(1);
- complaint — 20 working days, APPC Article 99, no extension;
- message, proposal, response, and inquiry — 15 working days, APPC Articles 87 + 76(1);
- inquiry under Law No. 401-V — 15 calendar days;
- local petition — 20 working days.

Counting starts on the day after registration. Arrivals after 18:00, on weekends, or on holidays are registered on the next working day. If the final day is non-working, the deadline moves forward. Holidays are editable in `data/holidays.json`.

`TODO_VERIFY_KURBAN_AIT_ENTER_MANUALLY` is intentional: the Kurban Ait date must be checked by a person each year and is not calculated programmatically.

An overdue case enters **“Deemed refusal — APPC Article 91(2)”**. The interface does not claim an administrative fine: Administrative Code Article 189 was removed on 1 July 2021. The risk tooltip explains the disciplinary track under the Law on Civil Service.

## Routing

`data/routing.json` distinguishes the operational assignee from the statutory deadline holder. For example, water-supply requests are routed both to the municipal enterprise “Kokshetau Su Arnasy” and to the Department of Housing and Utilities, Passenger Transport, and Highways; the private company “Kokshe Tazalyk” is not assigned an APPC deadline.

The saved-working-days metric is an **upper-bound estimate**, not measured savings: `3 working days × number of cases with determined competence`. Three days is the maximum forwarding period under APPC Article 65(1). Savings are zero for cases with undetermined competence.

Housing inspection / condominium management, power outages, and stray animals are intentionally routed to `UNDETERMINED — clarification required` until the Akimat confirms competence.

## Geography and data accuracy

Addresses are normalized and matched only against the local `data/streets.json`; there is no network geocoder. Coordinates are approximate demonstration points within Kokshetau, Krasny Yar, and Stantsionny. For a pilot, replace them with the Akimat’s address register or licensed 2GIS data.

The street list was assembled from public OpenStreetMap-, 2GIS-, and Yandex-derived sources **and is not an official street registry**. It must not be presented to Akimat staff as an authoritative source.

The offline map uses 1,253 locally rendered XYZ tiles from OpenStreetMap data for Kokshetau at zoom levels 11–15. Streets and the railway remain visible without network access; OpenStreetMap attribution is shown on the map. Schematic layers for Lake Kopa and the Shagalaly and Kylshakty rivers are preserved over the basemap.

## Required language review

**Every Kazakh-language, mixed-language, and Latin-script request must be read and corrected by Tokha before any demonstration.** The corpus contains 150 such records. This repository does not claim that generated Kazakh text has been reviewed by a native speaker. Kazakh draft replies also require language and legal proofreading.

## Structure

```text
app.py                         Streamlit app with four tabs
jauap/schema.py                domain dataclass models
jauap/corpus.py                one-time synthetic corpus generation
jauap/classify.py              classification and routing
jauap/deadline_engine.py       APPC calendar and deadlines
jauap/geo.py                   local address extraction and resolution
jauap/cluster.py               single-linkage clustering
jauap/risk.py                  transparent weighted risk
jauap/draft.py                 draft replies and notifications
jauap/llm.py                   Gemini-primary model boundary and cache
data/demo_corpus.json          250 source synthetic requests
data/demo_ground_truth.json    isolated reference labels for scoring only
data/demo_results.json         generated by scripts/freeze_demo.py from model responses
data/tiles/                    local offline Kokshetau tiles
```

This is a demonstration, not a production system. Authentication, a database, eOtinish or government APIs, mobile or public-facing workflows, Docker, CI, cloud deployment, speech recognition, model training, and an embedding database are intentionally out of scope.
