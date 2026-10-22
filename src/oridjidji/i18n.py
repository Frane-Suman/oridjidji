"""The single Croatian UI-string dictionary (the project specification).

Code and docs are English; every user-facing string lives here and only here.
Keys are stable identifiers referenced from library code and, later, the web UI.

Complexity justification: this file exceeds 300 lines because the project contract requires one
auditable dictionary for every Croatian UI sentence. Splitting it would weaken the central
"i18n-only visible copy" test without reducing any algorithmic complexity.
"""

from __future__ import annotations

from typing import Final

MESSAGES_HR: Final[dict[str, str]] = {
    "lang_rejected": (
        "Uneseni tekst nije prepoznat kao hrvatski (prepoznato: {lang}). "
        "Oriđiđi je kalibriran isključivo za hrvatski jezik i za druge jezike ne daje rezultat."
    ),
    "lang_uncertain_bcs": (
        "Napomena: pouzdano razlikovanje hrvatskog od srpskog i bosanskog jezika nije moguće; "
        "rezultat vrijedi za taj jezični kontinuum."
    ),
    "too_short": (
        "Tekst je prekratak za pouzdanu analizu (potrebno je najmanje {min_words} riječi)."
    ),
    "homoglyph_warning": (
        "Tekst sadrži znakove koji oponašaju latinicu (homoglife) — moguć pokušaj "
        "zaobilaženja detekcije. Analiza je provedena na normaliziranom tekstu."
    ),
    # --- verdicts (never an absolute claim, always a probability) -------
    #
    # These are the brandbook's headlines and they deliberately NEVER say
    # "human": a low score reported as "probably human" would read as clearance, and it is the
    # output most likely to be quoted at someone, which the model cannot support.
    # Every headline therefore reports a degree of MACHINE signal, or its absence.
    #
    # The `Verdict` enum is untouched; `LIKELY_HUMAN` simply renders as weak evidence
    # of machine writing rather than as evidence of a person.
    "verdict_likely_human": "Slabi znakovi strojnog pisanja",
    "verdict_inconclusive": "Neodlučno — nema dovoljno signala",
    "verdict_likely_ai": "Jaki znakovi strojnog pisanja",
    "verdict_inconclusive_help": (
        "Ovo nije pogreška ni kvar, nego uredan ishod: dokazi nisu dovoljno jaki za "
        "svrstavanje na bilo koju stranu. Postupak je namjerno tako postavljen — radije "
        "će reći „ne znam” nego pogriješiti."
    ),
    # --- the permanent ethics panel (furniture, never dismissible) ------
    "disclaimer_short": (
        "Rezultat je procjena vjerojatnosti, a ne dokaz. Ne smije se koristiti za "
        "ocjenjivanje, stegovne postupke ni bilo kakvu odluku o pojedincu."
    ),
    "limitations_title": "Ograničenja — pročitajte prije tumačenja rezultata",
    "limitation_not_evidence": (
        "Ovo nije dokaz. Nijedan detektor strojno generiranog teksta, pa ni ovaj, ne "
        "može utvrditi autorstvo. Rezultat je kalibrirana procjena i ništa više."
    ),
    "limitation_min_length": (
        "Ispod otprilike {min_words} riječi procjena nije pouzdana. Kratki tekstovi ne "
        "sadrže dovoljno signala i rezultat na njima nije upotrebljiv."
    ),
    "limitation_scope": (
        "Procjenjuje se samo početak teksta. Rezultat ne opisuje kasnije dijelove dužega "
        "teksta i ne može otkriti tekst u kojem je samo dio strojno napisan."
    ),
    "limitation_paraphrase": (
        "Parafraziranje i ljudsko dorađivanje strojnog teksta bitno smanjuju točnost. "
        "Tko želi izbjeći detekciju, u pravilu u tome uspije."
    ),
    "limitation_false_positive": (
        "Na ispitanim ljudskim tekstovima zabilježeni su lažno pozitivni nalazi, osobito "
        "u akademskom uzorku. Posljedice takve pogreške ne snose svi jednako."
    ),
    # ================= web UI (brandbook, verbatim) =====================
    # Verbatim from the brief: "Do not translate, improvise, or 'improve' the Croatian
    # microcopy." They live here rather than in the templates because the specification makes this
    # file the single source for every user-facing string.
    #
    # --- header & hero ---------------------------------------------
    "ui_tagline": "PROCJENA, NE PRESUDA",
    "ui_skip_link": "Prijeđi na glavni sadržaj",
    "ui_nav_label": "Glavna navigacija",
    "ui_nav_method": "Metoda",
    "ui_nav_analyse": "Analiza",
    "ui_nav_history": "Povijest",
    "ui_logo_alt": "Oriđiđi",
    "ui_hero_title": "Koliko je vjerojatno da je ovaj tekst napisala umjetna inteligencija?",
    "ui_hero_subline": (
        "Odaberi brzu BERTić procjenu ili sporiju višemetodnu analizu. "
        "Oba načina prikazuju kalibriranu procjenu i mogući neodlučan ishod."
    ),
    "ui_mode_basic": "Osnovno",
    "ui_mode_advanced": "Napredno",
    "ui_mode_label": "Način analize",
    "ui_mode_basic_description": "Brža analiza početka teksta uvježbanim BERTić klasifikatorom.",
    "ui_mode_advanced_description": (
        "Sporija istraživačka analiza koja spaja više metoda i prikazuje dodatne signale."
    ),
    "ui_additional_options": "Dodatne postavke",
    "ui_domain_label": "Vrsta teksta",
    "ui_domain_auto": "Automatski",
    "ui_domain_academic": "Akademski tekst",
    "ui_domain_news": "Vijest",
    "ui_domain_help": (
        "Odabir trenutačno organizira povijest analiza i ne mijenja izračun procjene."
    ),
    "ui_store_label": "Spremi uneseni tekst uz ovu analizu",
    "ui_store_help": (
        "Bez ovog odabira cijeli unos i ulomci iz naprednog prikaza ne spremaju se u lokalnu "
        "povijest. Rezultat i tehnički podaci ostaju u povijesti."
    ),
    # --- editor ----------------------------------------------------
    "ui_editor_label": "TEKST ZA ANALIZU",
    "ui_editor_placeholder": "Zalijepi tekst ovdje…",
    "ui_editor_analyse": "Analiziraj tekst",
    "ui_editor_clear": "Očisti",
    "ui_editor_wordcount": "{n_words} riječi · najmanje {min_words}",
    "ui_editor_remaining": "Dodaj još {remaining} riječi za pokretanje analize.",
    "ui_editor_ready": "Tekst je dovoljno dug za analizu.",
    "ui_editor_over_limit": "Ukloni {excess} riječi ili podijeli tekst na manje dijelove.",
    "ui_editor_scope_basic": (
        "Osnovna analiza procjenjuje samo početak teksta: prvih {encoder_tokens} tokena, "
        "približno {encoder_words} riječi. Ostatak dužega teksta ne ulazi u procjenu."
    ),
    "ui_editor_scope_advanced": (
        "Napredna analiza procjenjuje samo početak teksta: BERTić čita približno prvih "
        "{encoder_words} riječi, a statističke metode prvih {statistical_tokens} tokena, "
        "približno {statistical_words} riječi. Ostatak dužega teksta ne ulazi u procjenu."
    ),
    "ui_editor_scope_advanced_sampled": (
        "Napredna analiza ne procjenjuje cijeli dugi tekst: BERTić čita približno prvih "
        "{encoder_words} riječi, a statističke metode početak i kraj teksta."
    ),
    "ui_editor_scope_long": "Od {n_words} unesenih riječi procijenit će se samo početak.",
    "ui_editor_disabled_empty": "Unesi tekst za analizu.",
    "ui_editor_disabled_short": "Tekst još nema potreban broj riječi.",
    "ui_editor_disabled_long": "Tekst prelazi dopuštenu duljinu.",
    "ui_editor_shortcut": "Ctrl ili Cmd + Enter",
    "ui_editor_readonly": "Tekst je zaključan dok analiza traje.",
    "ui_history_not_found": "Analiza nije pronađena.",
    "ui_feedback_saved": "Povratna informacija je spremljena.",
    "ui_feedback_failed": "Povratna informacija nije spremljena. Pokušaj ponovno.",
    "ui_progress_pipeline_started": "Pokrećem analizu…",
    "ui_progress_agent_started": "Pokrećem detektor {agent}…",
    "ui_progress_agent_finished": "Detektor {agent} je završio.",
    "ui_progress_agent_failed": "Detektor {agent} nije dostupan; analiza se nastavlja.",
    "ui_progress_fusion_started": "Spajam dostupne signale…",
    "ui_progress_pipeline_finished": "Analiza je završila.",
    "ui_progress_interpretation": "Sažeto tumačim rezultat…",
    "ui_method_title": "Metoda i trenutačni sklop",
    "ui_method_intro": (
        "Oriđiđi u osnovnom načinu koristi uvježbani klasifikator, a u naprednom mu pridružuje "
        "statističke metode. Prikazuje kalibriranu procjenu; kada signali nisu dovoljni, "
        "ishod je neodlučan."
    ),
    "ui_method_pipeline_title": "Tijek analize",
    "ui_method_runtime_title": "Podaci učitani pri pokretanju",
    "ui_method_min_words": "Najmanji broj riječi",
    "ui_method_agents": "Dostupni detektori",
    "ui_method_bundle": "Inačice artefakata",
    "ui_method_none": "Nijedan model nije učitan; poslužitelj radi u sigurnom neodlučnom stanju.",
    "ui_method_basic_title": "Osnovna analiza",
    "ui_method_basic_body": (
        "BERTić čita početak normaliziranoga hrvatskog teksta i daje sirovi rezultat. Hrvatska kalibracija "
        "pretvara ga u postotak, a konformalni sloj određuje podržava li signal zaključak."
    ),
    "ui_method_advanced_title": "Napredna analiza",
    "ui_method_advanced_body": (
        "Napredni način spaja BERTić s dvjema statističkim metodama nad jezičnim modelima. "
        "Jezične značajke objašnjavaju stil teksta, ali ne ulaze u završni postotak."
    ),
    "ui_method_probability_title": "Postotak i neodlučan ishod",
    "ui_method_probability_body": (
        "Postotak je kalibrirana vjerojatnost strojnog pisanja, procijenjena na hrvatskim "
        "kalibracijskim podatcima. Oznaku rezultata određuje konformalni skup predviđanja "
        "kalibriran zasebno za svaki razred: ako skup sadrži samo jedan razred, prikazuje se "
        "pripadajuća oznaka, a ako sadrži oba razreda ili nijedan, rezultat je „Neodlučno”."
    ),
    "ui_method_runtime_states_title": "Stanja naprednog načina",
    "ui_method_runtime_states_body": (
        "Oznaka istraživačkog načina opisuje granice dokaza, ali ne onemogućuje pokretanje. "
        "Ako statistički dio zakaže, sučelje izričito prikazuje zasebnu osnovnu procjenu."
    ),
    "ui_method_data_title": "Korpusi i licence",
    "ui_method_data_body": (
        "Sustav je razvijen na hrvatskim akademskim i novinskim tekstovima te raznolikim "
        "strojno generiranim parovima. Izvori, licence i uloge zapisa vezani su uz inačicu skupa podataka."
    ),
    "ui_method_privacy_title": "Privatnost",
    "ui_method_privacy_body": (
        "Cijeli uneseni tekst zasebno se sprema samo ako korisnik to odabere. U naprednom "
        "načinu odgovor na zaslonu može sadržavati ulomke teksta, ali bez pristanka oni se ne "
        "spremaju u lokalnu povijest. Rezultat i tehnički podaci mogu se ondje zadržati."
    ),
    "ui_method_limitations_title": "Pristranosti i poznata ograničenja",
    "ui_method_limitations_body": (
        "Točnost ovisi o duljini, domeni, generatoru i naknadnom uređivanju. Na ispitanim "
        "ljudskim tekstovima postoje lažno pozitivni nalazi; druge skupine nisu zasebno provjerene."
    ),
    "ui_method_references_title": "Korištene metode",
    "ui_method_reference_encoder": "BERTić klasifikator hrvatskog teksta",
    "ui_method_reference_statistical": "Statističke metode usporedbe jezičnih modela",
    "ui_method_reference_conformal": "Klasično uvjetno konformalno predviđanje",
    "ui_method_runtime_identity": "Identiteti učitanih artefakata",
    "ui_method_step_preprocess": "Tekst se normalizira te se provjeravaju jezik i duljina.",
    "ui_method_step_agents": (
        "Dostupni detektori obrađuju početni odsječak istoga normaliziranog teksta; ostatak "
        "dužega teksta ne ulazi u procjenu."
    ),
    "ui_method_step_calibrate": "Rezultati se spajaju i kalibriraju samo učitanim hrvatskim artefaktima.",
    "ui_method_step_conformal": "Konformalni skup određuje je li dopušten zaključak ili neodlučan ishod.",
    "ui_evidence_title": "Dodatni prikazi signala",
    "ui_agents_title": "Dokazi pojedinih metoda",
    "ui_agent_name": "Metoda",
    "ui_agent_status": "Stanje",
    "ui_agent_score": "Sirovi rezultat",
    "ui_agent_direction": "Smjer",
    "ui_agent_contribution": "Doprinos fuziji",
    "ui_agent_latency": "Trajanje",
    "ui_agent_coverage": "Obuhvat tokena",
    "ui_agent_details": "Obrazloženje",
    "ui_agent_used": "Sudjeluje u procjeni",
    "ui_agent_explanation_only": "Samo objašnjenje",
    "ui_agent_ok": "dostupno",
    "ui_agent_failed": "nedostupno",
    "ui_agent_no_rationale": "Dodatno obrazloženje nije dostupno.",
    "ui_yes": "da",
    "ui_no": "ne",
    "ui_agent_direction_higher": "viši rezultat znači više strojnog signala",
    "ui_agent_direction_lower": "niži rezultat znači više strojnog signala",
    "ui_agent_operating_note": (
        "Samostalni hrvatski pragovi služe prikazu dokaza i ne određuju završni rezultat fuzije."
    ),
    "ui_agent_operating_target": "Ciljana stopa lažno pozitivnih",
    "ui_agent_operating_value": "Sirova granična vrijednost",
    "ui_agent_operating_fpr": "Izmjerena stopa lažno pozitivnih",
    "ui_agent_operating_tpr": "Izmjerena osjetljivost",
    "ui_linguistic_title": "Jezično objašnjenje",
    "ui_linguistic_body": (
        "Ove značajke opisuju stil i pomažu tumačenju. Ne sudjeluju u izračunu završne procjene."
    ),
    "ui_token_heatmap": "Signali po riječima",
    "ui_feature_contributions": "Doprinosi jezičnih značajki",
    "ui_token_numeric_show": "Prikaži brojčane vrijednosti",
    "ui_token_numeric_hide": "Sakrij brojčane vrijednosti",
    "ui_token_legend_low": "slabiji signal",
    "ui_token_legend_high": "jači signal",
    "ui_text_signal_title": "Tragovi u analiziranom tekstu",
    "ui_text_signal_help": (
        "Jače označene riječi više su utjecale na statističke metode u ovoj analizi. "
        "Boja pokazuje relativan trag unutar ovog teksta, a ne zasebnu procjenu autorstva riječi."
    ),
    "ui_text_signal_document_label": (
        "Analizirani tekst s četiri razine označavanja relativnih statističkih tragova"
    ),
    "ui_text_signal_word_title": "Relativna jačina traga: {weight} %",
    "ui_interpretation_title": "Sažeto tumačenje",
    "ui_interpretation_ai": (
        "Analiza nalazi izražen strojni obrazac, ali time ne dokazuje strojno autorstvo."
    ),
    "ui_interpretation_human": (
        "Analiza nalazi malo strojnog obrasca, ali time ne potvrđuje ljudsko autorstvo."
    ),
    "ui_interpretation_inconclusive": (
        "Signali nisu dovoljno jednoznačni za pouzdano razlikovanje ljudskog i strojnog teksta."
    ),
    "ui_interpretation_precise_ai": (
        "Procjena iznosi {estimate} %. Rezultat upućuje na veliku vjerojatnost strojnog pisanja."
    ),
    "ui_interpretation_precise_human": (
        "Procjena iznosi {estimate} %. Rezultat više odgovara ljudskom pisanju."
    ),
    "ui_interpretation_precise_inconclusive": (
        "Procjena iznosi {estimate} %. Rezultat nije dovoljno jasan za pouzdan zaključak."
    ),
    "ui_agent_interpretation_failed": (
        "Metoda nije završila pa nije utjecala na prikazanu procjenu."
    ),
    "ui_agent_interpretation_explanation": (
        "Ova metoda opisuje jezični stil, ali ne mijenja završni postotak."
    ),
    "ui_agent_interpretation_calibrated": (
        "Rezultat ove metode pretvoren je hrvatskom kalibracijom u prikazanu procjenu."
    ),
    "ui_agent_interpretation_raises": ("Ova je metoda povećala završnu procjenu strojnog signala."),
    "ui_agent_interpretation_lowers": ("Ova je metoda smanjila završnu procjenu strojnog signala."),
    "ui_agent_interpretation_neutral": "Ova metoda gotovo nije promijenila završnu procjenu.",
    "ui_agent_score_higher_explanation": (
        "Sirovi rezultat {score}: kod ove metode viša vrijednost više upućuje na strojno pisanje."
    ),
    "ui_agent_score_lower_explanation": (
        "Sirovi rezultat {score}: kod ove metode niža vrijednost više upućuje na strojno pisanje."
    ),
    "ui_agent_score_basic_explanation": (
        "Ta se vrijednost uspoređuje s hrvatskim primjerima kako bi se dobio prikazani postotak."
    ),
    "ui_agent_contribution_raises_explanation": (
        "Doprinos {contribution} povukao je zajednički rezultat prema strojnom pisanju; "
        "taj broj nije postotni bod."
    ),
    "ui_agent_contribution_lowers_explanation": (
        "Doprinos {contribution} povukao je zajednički rezultat prema ljudskom pisanju; "
        "taj broj nije postotni bod."
    ),
    "ui_agent_contribution_neutral_explanation": (
        "Doprinos {contribution} gotovo nije pomaknuo zajednički rezultat; taj broj nije postotni bod."
    ),
    "ui_agent_threshold_ai_explanation": (
        "U usporedbi s hrvatskim pragom {threshold}, postavljenim za cilj {target_fpr} % "
        "lažno označenih ljudskih tekstova, rezultat je na strojnoj strani praga."
    ),
    "ui_agent_threshold_human_explanation": (
        "U usporedbi s hrvatskim pragom {threshold}, postavljenim za cilj {target_fpr} % "
        "lažno označenih ljudskih tekstova, rezultat je na ljudskoj strani praga."
    ),
    "ui_agent_technical_context": "Kako metoda računa signal",
    # --- result ----------------------------------------------------
    "ui_result_label": "REZULTAT",
    "ui_new_analysis": "Novi tekst",
    "ui_result_basis": "Procjena se temelji na {used} od {total} detektora.",
    "ui_scope_title": "Analizirani dio teksta",
    "ui_scope_whole": "Procijenjen je cijeli tekst ({n_words} riječi).",
    "ui_scope_basic": (
        "Procijenjen je samo početak teksta: prvih {encoder_words} od {n_words} riječi. "
        "Ostatak teksta nije analiziran."
    ),
    "ui_scope_advanced": (
        "Procijenjen je samo početak teksta od {n_words} riječi: BERTić je pročitao prvih "
        "{encoder_words}, a statističke metode prvih {statistical_words} riječi. Ostatak "
        "teksta nije analiziran."
    ),
    "ui_scope_advanced_statistical_whole": (
        "Statističke metode pročitale su cijeli tekst od {n_words} riječi, a BERTić samo "
        "prvih {encoder_words} riječi."
    ),
    "ui_scope_advanced_sampled": (
        "Procijenjen je samo dio teksta od {n_words} riječi: BERTić je pročitao prvih "
        "{encoder_words}, a statističke metode početak i kraj teksta."
    ),
    "ui_scope_unknown": ("Procijenjen je dio teksta od {n_words} riječi; obuhvat nije zabilježen."),
    "ui_result_inconclusive_body": (
        "Procjena ne podupire nijednu oznaku: analizirani dio teksta ne nalikuje dovoljno "
        "ni ljudskim ni strojnim tekstovima na kojima je sustav kalibriran. Tekst dodan iza "
        "analiziranoga početka ne mijenja procjenu."
    ),
    "ui_result_copy": "Kopiraj izvještaj",
    "ui_result_how": "Kako ovo radi",
    "ui_result_executed_mode": "Izvedeni način: {mode}",
    "ui_result_requested_mode": "Zatraženi način: {mode}",
    "ui_result_prediction_set": "Konformalni skup: {labels}",
    "ui_result_prediction_single": "Skup predviđanja podržava prikazani zaključak.",
    "ui_result_prediction_ambiguous": (
        "Skup predviđanja uključuje obje mogućnosti, pa je rezultat neodlučan."
    ),
    "ui_result_prediction_empty": (
        "Tekst ne nalikuje dovoljno ni jednoj kalibracijskoj klasi, pa se sustav suzdržava."
    ),
    "ui_result_assessment_count": "Procjena se temelji na {used} od {total} metoda.",
    "ui_result_warnings": "Upozorenja",
    "ui_result_copied": "Izvještaj je kopiran.",
    "ui_result_copy_failed": "Izvještaj nije moguće kopirati. Označi ga i kopiraj ručno.",
    "ui_result_rail_label": ("Procjena {estimate} posto, ishod: {verdict}."),
    "ui_result_scale_start": "0 %",
    "ui_result_scale_end": "100 %",
    "ui_runtime_research_title": "Istraživačka napredna procjena",
    "ui_runtime_fallback_title": "Prikazana je osnovna procjena",
    "ui_runtime_unavailable_title": "Napredni način trenutačno nije dostupan",
    "ui_runtime_unavailable_body": (
        "Odabir ostaje vidljiv. Provjeri jesu li svi napredni artefakti učitani pa pokušaj ponovno."
    ),
    # --- progress --------------------------------------------------
    # The ONLY progress string the frontend owns. Every other step label arrives on
    # the SSE event, because a label the frontend invents is a claim about a pipeline
    # it cannot see.
    "ui_progress_preparing": "Pripremam analizu…",
    # --- empty & error states --------------------------------------
    "ui_empty_result": "Rezultat će se pojaviti ovdje. Zalijepi tekst s lijeva i pokreni analizu.",
    "ui_error_no_text": "Nema teksta za analizu.",
    "ui_error_too_short": "Tekst je prekratak za analizu.",
    "ui_error_too_long": "Tekst je predug za jednu analizu. Podijeli ga na manje dijelove.",
    "ui_error_model_unavailable": (
        "Model se nije učitao. Provjeri je li datoteka modela na mjestu i pokušaj ponovno."
    ),
    "ui_error_analysis_failed": "Analiza nije dovršena. Pokušaj ponovno.",
    "ui_error_stream_dropped": "Veza s analizom je prekinuta prije kraja. Pokreni analizu ponovno.",
    "ui_error_invalid_request": "Unos ili odabrane postavke nisu prihvaćeni.",
    "ui_error_rate_limited": "Previše analiza čeka na izvođenje. Pričekaj i pokušaj ponovno.",
    "ui_error_too_large": "Zahtjev je prevelik. Skrati tekst i pokušaj ponovno.",
    "ui_retry": "Pokušaj ponovno",
    # --- agent legend ------------------------------------------------------
    "ui_agent_unavailable": "nedostupno",
    "ui_agent_state_pending": "čeka",
    "ui_agent_state_started": "u tijeku",
    "ui_agent_state_finished": "dovršeno",
    "ui_agent_state_failed": "nedostupno",
    # --- history -----------------------------------------------------------
    "ui_history_title": "Povijest analiza",
    "ui_history_unknown": "nije navedeno",
    "ui_history_intro": "Lokalno spremljeni rezultati i tehnički podaci izvođenja.",
    "ui_history_filters": "Filtri",
    "ui_history_filter_all": "Sve",
    "ui_history_filter_mode": "Način",
    "ui_history_filter_verdict": "Ishod",
    "ui_history_filter_domain": "Vrsta teksta",
    "ui_history_filter_status": "Stanje izvođenja",
    "ui_history_apply": "Primijeni filtre",
    "ui_history_export": "Preuzmi CSV",
    "ui_history_loading": "Učitavam povijest…",
    "ui_history_empty": "Nema analiza koje odgovaraju odabranim filtrima.",
    "ui_history_open": "Otvori detalje",
    "ui_history_run": "Analiza {run_id}",
    "ui_history_created": "Vrijeme",
    "ui_history_mode": "Način",
    "ui_history_domain": "Vrsta teksta",
    "ui_history_verdict": "Ishod",
    "ui_history_status": "Stanje",
    "ui_history_words": "Broj riječi",
    "ui_history_probability": "Procjena strojnog pisanja",
    "ui_history_text_title": "Spremljeni tekst",
    "ui_history_text_absent": (
        "Cijeli uneseni tekst i tekstualni ulomci nisu spremljeni; rezultat i tehnički podaci "
        "ostaju u povijesti."
    ),
    "ui_history_feedback_title": "Povratna informacija",
    "ui_history_feedback_label": "Očekivana oznaka",
    "ui_history_feedback_human": "ljudski tekst",
    "ui_history_feedback_ai": "strojni tekst",
    "ui_history_feedback_edited": "uređeni ili mješoviti tekst",
    "ui_history_feedback_unknown": "ne znam",
    "ui_history_feedback_comment": "Napomena",
    "ui_history_feedback_submit": "Spremi povratnu informaciju",
    "ui_history_timings_title": "Trajanje analize",
    "ui_history_timing_total": "Ukupno trajanje",
    "ui_history_timing_agent": "Trajanje metode",
    "ui_history_runtime_title": "Tehnički identitet izvođenja",
    "ui_history_back": "Natrag na povijest",
    "ui_history_failed": "Povijest nije moguće učitati.",
    "ui_history_status_started": "u tijeku",
    "ui_history_status_succeeded": "dovršeno",
    "ui_history_status_refused": "odbijeno",
    "ui_history_status_failed": "neuspjelo",
    # --- development gallery ----------------------------------------------
    "ui_review_title": "Pregled razvojnih stanja",
    "ui_review_banner": (
        "RAZVOJNI PREGLED — prikazani brojevi su izmišljeni i ne potječu ni iz jedne analize."
    ),
    "ui_review_loading": "Izvođenje u tijeku",
    "ui_review_empty": "Početno stanje",
    "ui_review_research": "Istraživački napredni rezultat",
    "ui_review_fallback": "Povratak na osnovni način",
    "ui_review_unavailable": "Napredni način nije dostupan",
    "ui_review_errors": "Pogreške i odbijeni zahtjevi",
    "ui_notice_statistical": "Ovo je statistička procjena, a ne dokaz.",
    "ui_notice_no_sanction": "Ne koristi je kao osnovu za optužbu, ocjenu ili sankciju.",
    "ui_notice_language_bias": (
        "Na ispitanim ljudskim tekstovima zabilježeni su lažno pozitivni nalazi, osobito u akademskoj domeni."
    ),
    # One label per detector band. Served rather than held by the frontend (12.5):
    # the UI must not know which detectors exist, only how to draw the ones it is told
    # about. Named by what it does, not by the checkpoint it happens to load.
    "agent_encoder": "Uvježbani klasifikator",
    # A4 (M1/M6): the Croatian feature set as a scoring agent. Named for what a reader can
    # verify — it looks at how the text is written — rather than for the 37 extractors behind
    # it. Missing this key crashed `create_app` the first time A4 was served, because every
    # serving test used a stub subclassing EncoderDetector and inherited the encoder's label.
    "agent_linguistic": "Jezične značajke",
    # The Advanced-mode agents (M5). Named for what they measure rather than for the
    # papers they come from — a reader of the UI has not read Hans et al.
    "agent_binoculars": "Dvostruki obzervator",
    "agent_binoculars_v4": "Dvostruki obzervator na granicama riječi",
    "agent_fast_detect_gpt": "Zakrivljenost vjerojatnosti",
    "advanced_degraded_notice": (
        "Statistički dio naprednog načina nije uspio, pa je prikazana zasebno "
        "kalibrirana procjena osnovnog BERTić modela."
    ),
    "advanced_research_notice": (
        "Napredna procjena koristi eksperimentalnu fuziju kalibriranu na GPU-u. "
        "Prijenos njezinih postotaka na CPU nije potvrđen, pa rezultat služi "
        "istraživačkoj demonstraciji, a ne validiranoj procjeni."
    ),
    # --- calibration state (2.1, 2.3) --------------------------------------
    # Shown whenever a detector has run but nothing has mapped its score onto observed
    # frequencies. Without a fitted calibrator and its conformal quantiles a score is not a
    # probability, so the result abstains and says why, rather than presenting a raw network
    # output as one.
    "uncalibrated_notice": (
        "Detektor je izračunao rezultat, ali procjena još nije kalibrirana na "
        "hrvatskim podacima, pa se ne prikazuje kao vjerojatnost. Sirovi rezultat vidljiv "
        "je u naprednom prikazu."
    ),
    # Shown when the conformal prediction set comes back EMPTY: the text is nonconforming to
    # both classes at once, so it is unlike anything the detector was calibrated on. Distinct
    # from `uncalibrated_notice`, which says no calibration exists, and from an ambiguous
    # result, whose set holds BOTH labels. All three abstain; conformal.py keeps them
    # separable precisely so the UI does not flatten them into one apology (2.1, 2.3).
    "atypical_notice": (
        "Tekst ne nalikuje ni ljudskim ni strojnim primjerima na kojima je detektor "
        "kalibriran, pa se sustav suzdržava od procjene. To nije nalaz o autorstvu nego o "
        "tome da je tekst izvan područja na kojem je alat mjeren."
    ),
    "limitation_never_discipline": (
        "Nikada za ocjenjivanje ni stegovne postupke. Alat je istraživački i obrazovni; "
        "upotreba protiv pojedinca nije predviđena namjena."
    ),
}

# The panel, in display order. A tuple rather than a loose list of keys so the UI
# cannot quietly render a subset: the specification makes this furniture, not a dismissible notice.
LIMITATION_KEYS: Final[tuple[str, ...]] = (
    "limitation_not_evidence",
    "limitation_min_length",
    "limitation_scope",
    "limitation_paraphrase",
    "limitation_false_positive",
    "limitation_never_discipline",
)

# The notice, verbatim and in order. Three sentences, always all three: the
# brandbook renders it on every result screen without exception, and a partial notice
# is the failure mode the whole panel exists to prevent.
NOTICE_HR: Final[tuple[str, ...]] = (
    MESSAGES_HR["ui_notice_statistical"],
    MESSAGES_HR["ui_notice_no_sanction"],
    MESSAGES_HR["ui_notice_language_bias"],
)


VERDICT_MESSAGE_KEYS: Final[dict[str, str]] = {
    "LIKELY_HUMAN": "verdict_likely_human",
    "INCONCLUSIVE": "verdict_inconclusive",
    "LIKELY_AI": "verdict_likely_ai",
}


def get_message(key: str, **kwargs: object) -> str:
    """Return the Croatian message for ``key``, formatted with ``kwargs``."""
    return MESSAGES_HR[key].format(**kwargs)
