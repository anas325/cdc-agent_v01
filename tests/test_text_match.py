"""Unit tests for the deterministic "have we already asked this?" matcher.

The thresholds here are calibrated against a real failing benchmark run
(bench_20260806_100245 / cdc_003_ecommerce), where the same cart-retention
question went out eleven times. Both directions matter: catching the repeats,
and *not* suppressing the neighbouring questions that were legitimately distinct.
"""

from __future__ import annotations

from src.utils.text_match import (
    containment,
    content_tokens,
    is_near_duplicate,
    is_too_vague,
    first_near_duplicate,
    strip_entity_ids,
)

# The question that was actually asked first, and its later restatements.
ORIGINAL = (
    "Dans le cahier des charges, l'Assumption (ctx_7f3e35b23286) indique que le panier est "
    "conservé 24 h après une erreur de paiement, tandis que la réponse utilisateur "
    "(ctx_004f1430d9b3) indique 30 jours. Quelle est la durée de conservation souhaitée "
    "pour le panier après une erreur de paiement ?"
)
RESTATEMENTS = [
    "Quelle est la durée de conservation souhaitée pour le panier après une erreur de paiement ?",
    "Quelle durée de conservation du panier après une erreur de paiement doit être retenue, "
    "30 jours ou 24 heures ?",
    "Le cahier des charges indique que le panier est conservé 30 jours après un paiement refusé "
    "(ctx_7a386fc54817) et 24 h après une erreur de paiement (ctx_7f3e35b23286). Quelle durée "
    "doit être retenue ?",
]
# Asked in the same run, about other things — must survive.
DISTINCT = [
    "Quelles sont les exclusions et les phases de déploiement prévues pour le périmètre technique ?",
    "Quel est le budget maximal alloué à la phase 1 ?",
    "Pouvez-vous fournir le schéma de base de données, les entités et leurs relations ?",
    "Quelles exigences de sécurité supplémentaires (authentification, protection des données, "
    "conformité PCI DSS) doivent être intégrées aux critères d'acceptation des livrables techniques ?",
]


def test_restatements_of_the_same_question_are_caught():
    for restatement in RESTATEMENTS:
        assert is_near_duplicate(restatement, ORIGINAL), restatement


def test_distinct_questions_are_not_suppressed():
    for question in DISTINCT:
        assert not is_near_duplicate(question, ORIGINAL), question
    # ...nor against each other.
    for i, a in enumerate(DISTINCT):
        for b in DISTINCT[i + 1 :]:
            assert not is_near_duplicate(a, b), (a, b)


def test_entity_ids_do_not_make_two_restatements_look_different():
    """Ids are re-minted every turn; they must not count as content."""
    with_ids = "Le panier est conservé 30 jours (ctx_9eb2a611485d) ou 24 heures (ctx_7f3e35b23286) ?"
    without = "Le panier est conservé 30 jours (ctx_aaaaaaaaaaaa) ou 24 heures (ctx_bbbbbbbbbbbb) ?"
    assert content_tokens(with_ids) == content_tokens(without)
    assert "ctx" not in strip_entity_ids(with_ids)


def test_accents_are_folded():
    assert content_tokens("Quel délai de conservation ?") == content_tokens("Quel delai de conservation ?")


def test_a_single_shared_content_word_is_not_a_duplicate():
    """One word names a topic, not a question — decline rather than suppress."""
    assert not is_near_duplicate("blocking", "blocking")
    assert not is_near_duplicate("Quel budget ?", "Quel budget ?")


def test_short_questions_need_an_exact_match():
    """Two or three content words: a ratio means nothing, so demand equality.

    Matching is on whole words — there is no stemming — so "délai" and "délais"
    are different content. That is deliberate: at this length, guessing wrong
    silently drops a question the author never got to answer.
    """
    assert is_near_duplicate("Quel délai de livraison ?", "Quels sont les délai de livraison ?")
    assert not is_near_duplicate("Quel délai de livraison ?", "Quel délai de paiement ?")


def test_is_too_vague_flags_a_question_narrowed_to_nothing():
    assert is_too_vague("Quelle durée doit être retenue ?")
    assert not is_too_vague(RESTATEMENTS[1])


def test_first_near_duplicate_returns_the_earliest_match():
    previous = [*DISTINCT, ORIGINAL, RESTATEMENTS[0]]
    assert first_near_duplicate(RESTATEMENTS[1], previous) == len(DISTINCT)
    assert first_near_duplicate("Quel est le taux de TVA applicable aux commandes ?", previous) is None


def test_containment_is_symmetric_and_bounded():
    assert containment(ORIGINAL, ORIGINAL) == 1.0
    assert containment(RESTATEMENTS[0], ORIGINAL) == containment(ORIGINAL, RESTATEMENTS[0])
    assert containment("", ORIGINAL) == 0.0
