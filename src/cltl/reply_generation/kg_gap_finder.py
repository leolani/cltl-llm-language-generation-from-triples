#!/usr/bin/env python3
"""
kg_gap_finder.py
================

Queries an RDF knowledge graph for "knowledge gaps", in the spirit of the
Leolani Brain (leolani/cltl-knowledgerepresentation), which generates
"curiosity" thoughts based on gaps in the graph.

Background
----------
cltl-knowledgerepresentation builds an episodic RDF graph (backed by GraphDB,
queried with SPARQL) and derives "thoughts" from it, one category of which is
knowledge-graph gaps: cases where an entity is missing information that
similar entities in the graph do have. This script re-implements that idea
generically, so it works against:

  1. A live SPARQL endpoint (e.g. a running GraphDB "sandbox" repository, the
     default used by the cltl-knowledgerepresentation examples), or
  2. A local RDF file (turtle/xml/n3/json-ld) loaded with rdflib.

Gaps are reported at the TYPE level for the *pattern* (which class/predicate
combination is affected, and how many peers share the expectation), while
still naming the exact affected entity. Each finding describes a pattern
("instances of Person are missing hasName in 4/10 cases") and, for gap kinds
B, D and E, one row per affected subject naming exactly which entity has the
gap, rather than aggregating into a single group with a handful of examples.
This makes the report useful both for spotting systematic modeling gaps in
the ontology/data, and for drilling straight down to the specific entity to
fix.

It reports five kinds of gaps:

  A. Untyped entities        - resources used as subjects that have no
                                rdf:type, grouped by the *signature* of
                                predicates they do have (a proxy for "what
                                type they probably are").
  B. Predicate gaps          - for each rdf:type class, predicates that a
                                clear majority of instances have, but some
                                instances lack; one row per (class,
                                predicate, subject) with a peer-coverage
                                count (the "I know X about others like you,
                                but not about you" gap).
  C. Dangling references     - objects that are referenced (as the object of
                                a triple) but that have no outgoing triples of
                                their own in the graph, grouped by the
                                (referencing type, predicate) that points at
                                them, i.e. "Person hasFriend points at 12
                                references we know nothing further about".
  D. Predicate-object gaps,  - for each rdf:type class, (predicate, object-
     TYPE level                TYPE) facts that a majority of instances
                                share; one row per (class, predicate,
                                object-type, subject) with a peer-coverage
                                count (finer grained than B, coarser than E:
                                "most people like you have a hasPet of type
                                Dog, but a few of you specifically don't" -
                                any Dog satisfies the expectation, regardless
                                of which one).
  E. Predicate-object gaps,  - the same idea as D, but objects are compared
     INSTANCE level            by exact identity instead of by type: "most
                                people like you have hasPet=Fido, but a few
                                of you specifically don't" - a different dog
                                does NOT satisfy the expectation. Tends to
                                fragment into many individual-specific gaps
                                when a predicate points at many distinct
                                instances of the same type.

Usage
-----
    # Against a running GraphDB / any SPARQL endpoint:
    python kg_gap_finder.py --endpoint http://localhost:7200/repositories/sandbox

    # Against a local RDF file:
    python kg_gap_finder.py --file mybrain.trig --format trig

    # Only show gaps for a specific class (also covers its rdfs:subClassOf
    # descendants), and require >=70% majority:
    python kg_gap_finder.py --endpoint http://localhost:7200/repositories/sandbox \\
        --class-uri http://cltl.nl/leolani/n2mu/Person --threshold 0.7

    # Restrict predicate gaps (B, D and E) to one specific subject instead of
    # aggregating over every instance of its class:
    python kg_gap_finder.py --file mybrain.trig --subject-uri http://example.org/Piek

    # Show more example instances per gap group for A and C (default 5):
    python kg_gap_finder.py --file mybrain.trig --examples 10

    # Save the full report as JSON:
    python kg_gap_finder.py --file mybrain.trig --json report.json

Requires: rdflib  (pip install rdflib)
For remote SPARQL endpoints rdflib's built-in SPARQLWrapper support is used
(pip install rdflib sparqlwrapper).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from rdflib import Graph, URIRef
    from rdflib.plugins.stores import sparqlstore
except ImportError:
    print("This script needs rdflib. Install it with: pip install rdflib sparqlwrapper")
    sys.exit(1)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DEFAULT_EXAMPLE_LIMIT = 5


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #

def load_graph_from_endpoint(endpoint: str) -> Graph:
    """Connect read-only to a remote SPARQL endpoint (e.g. GraphDB)."""
    store = sparqlstore.SPARQLStore(endpoint)
    graph = Graph(store=store)
    return graph


def load_graph_from_file(path: str, fmt: str = None) -> Graph:
    graph = Graph()
    graph.parse(path, format=fmt)
    return graph


def run_query(graph: Graph, query: str) -> List[Dict]:
    """Run a SPARQL SELECT query and return rows as plain dicts of strings.

    Unbound variables (e.g. from an OPTIONAL clause) are kept as None rather
    than being stringified, so callers can distinguish "unbound" from the
    literal string "None".
    """
    results = graph.query(query)
    rows = []
    for row in results:
        rows.append({str(v): (str(row[v]) if row[v] is not None else None) for v in results.vars})
    return rows


# --------------------------------------------------------------------------- #
# A. Untyped entities, grouped by predicate signature
# --------------------------------------------------------------------------- #

UNTYPED_QUERY = """
SELECT ?entity ?p WHERE {
    ?entity ?p ?o .
    FILTER NOT EXISTS { ?entity a ?type }
    FILTER(isIRI(?entity))
}
LIMIT %d
"""


def find_untyped_entities(
    graph: Graph, row_limit: int = 5000, example_limit: int = DEFAULT_EXAMPLE_LIMIT
) -> List[Dict]:
    """
    Find entities with no rdf:type, then group them by the *set* of
    predicates they're used with. Entities that share an identical predicate
    signature are almost certainly missing the same type declaration, so
    this surfaces "there's a whole class of untyped entity here" instead of
    a flat list of individual untyped resources.
    """
    rows = run_query(graph, UNTYPED_QUERY % row_limit)

    entity_predicates: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        entity_predicates[r["entity"]].add(r["p"])

    signature_groups: Dict[frozenset, List[str]] = defaultdict(list)
    for entity, preds in entity_predicates.items():
        signature_groups[frozenset(preds)].append(entity)

    results = []
    for sig, instances in signature_groups.items():
        instances_sorted = sorted(instances)
        results.append(
            {
                "predicate_signature": sorted(sig),
                "instance_count": len(instances_sorted),
                "example_entities": instances_sorted[:example_limit],
            }
        )
    results.sort(key=lambda g: -g["instance_count"])
    return results


# --------------------------------------------------------------------------- #
# B. Predicate gaps per class ("others like you have this, some of you don't")
# --------------------------------------------------------------------------- #

CLASSES_QUERY = """
SELECT DISTINCT ?type WHERE { ?s a ?type . FILTER(isIRI(?type)) }
"""

INSTANCES_QUERY = """
SELECT DISTINCT ?instance WHERE { ?instance a <%s> . FILTER(isIRI(?instance)) }
LIMIT %d
"""

# Like INSTANCES_QUERY, but also matches instances of (transitive) rdfs:subClassOf
# descendants of <%s>, not just exact-type instances. Used when a class_filter is
# given explicitly, so e.g. --class-uri .../Person also picks up .../Student,
# .../Employee, etc. The subClassOf* path includes the zero-length case, so
# exact-type instances of <%s> itself are still matched too.
SUBCLASS_INSTANCES_QUERY = """
SELECT DISTINCT ?instance WHERE {
    ?instance a ?actualType .
    ?actualType <http://www.w3.org/2000/01/rdf-schema#subClassOf>* <%s> .
    FILTER(isIRI(?instance))
}
LIMIT %d
"""

PREDICATES_FOR_INSTANCE_QUERY = """
SELECT DISTINCT ?p WHERE { <%s> ?p ?o . }
"""

TYPES_FOR_SUBJECT_QUERY = """
SELECT DISTINCT ?type WHERE { <%s> a ?type . FILTER(isIRI(?type)) }
"""

PREDICATE_OBJECT_FOR_INSTANCE_QUERY = """
SELECT DISTINCT ?p ?o (isIRI(?o) AS ?oIsIRI) WHERE { <%s> ?p ?o . FILTER(!isBlank(?o)) }
"""

ALL_TRIPLES_FOR_SUBJECT_QUERY = """
SELECT ?p ?o (isIRI(?o) AS ?oIsIRI) WHERE { <%s> ?p ?o . }
"""


def _triples_of(graph: Graph, uri: str, cache: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Every (predicate, object) triple for a URI - i.e. everything already
    known about it, rdf:type included - memoized per call to a find_*
    function. Used to attach full context to a gap row: alongside the one
    fact a subject is missing, show what IS on record for it.

    Each returned dict also carries an internal "_object_is_iri" flag (only
    an IRI can meaningfully be the *subject* of further triples, so this is
    what _object_triples_of() uses to decide which objects to follow one
    more hop) - strip it before exposing rows to callers outside this
    module, e.g. with {k: v for k, v in t.items() if not k.startswith("_")}.
    """
    if uri not in cache:
        cache[uri] = sorted(
            (
                {"predicate": r["p"], "object": r["o"], "_object_is_iri": r["oIsIRI"] == "true"}
                for r in run_query(graph, ALL_TRIPLES_FOR_SUBJECT_QUERY % uri)
            ),
            key=lambda t: (t["predicate"], t["object"]),
        )
    return cache[uri]


def _strip_internal(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop the internal "_object_is_iri" bookkeeping field before exposing triples in a report row."""
    return [{"predicate": t["predicate"], "object": t["object"]} for t in triples]


def _types_of(graph: Graph, uri: str, cache: Dict[str, List[str]]) -> List[str]:
    """rdf:type(s) of a URI, memoized per call to a find_* function."""
    if uri not in cache:
        cache[uri] = [r["type"] for r in run_query(graph, TYPES_FOR_SUBJECT_QUERY % uri)]
    return cache[uri]


def _object_triples_of(
    graph: Graph,
    subject_triples: List[Dict[str, str]],
    triples_cache: Dict[str, List[Dict[str, str]]],
    types_cache: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    """
    One more hop out: for every object in `subject_triples` that is itself
    an IRI AND an instance (has its own rdf:type - so a data entity, not a
    class/schema/predicate URI referenced only as a value), fetch its own
    outgoing triples - i.e. "what do we know about the things this subject
    points at". Each object is only expanded once even if several
    subject_triples share it. Literal objects, and untyped IRIs (e.g. class
    URIs used as an rdf:type value), are skipped.
    """
    result = []
    seen_objects = set()
    for t in subject_triples:
        o = t["object"]
        if not t["_object_is_iri"] or o in seen_objects:
            continue
        seen_objects.add(o)
        if not _types_of(graph, o, types_cache):
            continue  # not an instance - nothing to follow
        for ot in _triples_of(graph, o, triples_cache):
            result.append({"subject": o, "predicate": ot["predicate"], "object": ot["object"]})
    result.sort(key=lambda t: (t["subject"], t["predicate"], t["object"]))
    return result


def find_predicate_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
) -> List[Dict]:
    """
    For every class, compute which predicates are used by >= `threshold`
    fraction of its instances ("expected" predicates), then, per (class,
    predicate), report every instance missing it as its own row - e.g.
    "Person / hasOccupation / subject=.../Alice (known for 9/12 peers)".
    Each row carries the group's `missing_count`/`total_instances`/
    `peer_coverage` for context, plus a `subject` field naming exactly which
    instance the row is about, so the gap is drillable without a separate
    examples list. Each row also carries `subject_triples`: every other
    (predicate, object) fact already known about that subject (rdf:type
    included), so the row is self-contained context for "what IS known
    about this one, and what's missing" rather than just the latter. And
    `object_triples`: one more hop out - every triple whose subject is an
    IRI object from `subject_triples` that is itself an instance (e.g. if
    the subject worksAt some Organization, `object_triples` includes that
    Organization's own facts; a class URI like the subject's own rdf:type
    value is not followed, since it isn't itself typed).

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class - e.g. class_filter=".../Person" pulls in
    instances of ".../Student" or ".../Employee" too, not just exact-type
    ".../Person" instances. Without class_filter (every declared class is
    analyzed), each class is still matched by exact rdf:type only.

    If `subject_filter` is given, only gaps touching that one subject are
    returned (but the majority is still computed over ALL peers in its
    class(es), consistent with find_predicate_object_gaps() /
    find_predicate_object_instances_gaps()).
    """
    if class_filter:
        classes = [class_filter]
    else:
        classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})

    instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}
    object_types_cache: Dict[str, List[str]] = {}

    gaps = []
    for cls in classes:
        instances = [
            r["instance"]
            for r in run_query(graph, instances_query % (cls, max_instances_per_class))
        ]
        if len(instances) < 2:
            continue  # nothing to compare against

        predicate_counts = defaultdict(int)
        instance_predicates: Dict[str, Set[str]] = {}

        for inst in instances:
            preds = {r["p"] for r in run_query(graph, PREDICATES_FOR_INSTANCE_QUERY % inst)}
            preds.discard(RDF_TYPE)
            instance_predicates[inst] = preds
            for p in preds:
                predicate_counts[p] += 1

        n = len(instances)
        expected = {p for p, c in predicate_counts.items() if c / n >= threshold}
        if not expected:
            continue

        predicate_missing_instances: Dict[str, List[str]] = defaultdict(list)
        for inst, preds in instance_predicates.items():
            if subject_filter and inst != subject_filter:
                continue
            for p in expected - preds:
                predicate_missing_instances[p].append(inst)

        for p, missing_instances in predicate_missing_instances.items():
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{predicate_counts[p]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# D. Predicate-object gaps, TYPE level ("peers have this fact about a peer of
#    this TYPE, some of you don't") - per class, per (predicate, object-type)
# --------------------------------------------------------------------------- #

def find_predicate_object_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
) -> List[Dict]:
    """
    For every class (or just `class_filter` if given), find (predicate,
    object-TYPE) combinations that a majority of instances of that class
    share, then, per (class, predicate, object-type), report every instance
    lacking a fact with an object of that type as its own row - even if it
    has some other value for the same predicate. Each row carries the
    group's `missing_count`/`total_instances`/`peer_coverage` for context,
    plus a `subject` field naming exactly which instance the row is about,
    `subject_triples`: every other (predicate, object) fact already known
    about that subject (rdf:type included), and `object_triples`: one more
    hop out - every triple whose subject is an IRI object from
    `subject_triples` that is itself an instance (has its own rdf:type).

    Object IRIs are compared by their rdf:type(s) rather than by their exact
    identity, e.g. "most Lunches have a hasPatient of type Wine" instead of
    "most Lunches have hasPatient=wine1" - two different Wine individuals
    (wine1, wine2) count as satisfying the same expectation, so this doesn't
    fragment into one gap per distinct individual the way instance-level
    comparison does. An object IRI with no declared rdf:type of its own falls
    back to being compared by its exact identity (nothing more specific is
    known about it). Literal objects have no separate "type" from their
    value, so they're still compared by exact literal value, same as before.
    An object with multiple rdf:types contributes one (predicate, type) pair
    per type.

    This is a finer-grained sibling of find_predicate_gaps(): that function
    flags a missing predicate in general (e.g. "some of you have no
    hasHobby"), while this one flags a specific missing predicate-object-type
    pair (e.g. "most people like you have a hasHobby of type OutdoorActivity,
    but 2 of you don't have that").

    For the exact-object-identity version of this check, see
    find_predicate_object_instances_gaps().

    Pairs and their peer counts are computed once per class and then reused
    for every instance in that class, rather than re-querying peers per
    subject.

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class (same as find_predicate_gaps()) - e.g.
    class_filter=".../Person" pulls in instances of ".../Student" etc. too.

    If `subject_filter` is given, only gaps touching that one subject are
    returned (but the majority is still computed over ALL peers in its
    class(es), consistent with the original single-subject behaviour) - in
    that case every returned row's "subject" will be that one subject.
    Blank-node objects are excluded since they have no stable identity to
    compare across instances.
    """
    if class_filter:
        classes = [class_filter]
    else:
        classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})

    instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
    object_types_cache: Dict[str, List[str]] = {}
    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}

    gaps = []
    for cls in classes:
        instances = [
            r["instance"]
            for r in run_query(graph, instances_query % (cls, max_instances_per_class))
        ]
        if len(instances) < 2:
            continue  # nothing to compare against

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        instance_pairs: Dict[str, Set[Tuple[str, str]]] = {}

        for inst in instances:
            pairs = set()
            for r in run_query(graph, PREDICATE_OBJECT_FOR_INSTANCE_QUERY % inst):
                if r["p"] == RDF_TYPE:
                    continue
                if r["oIsIRI"] == "true":
                    obj_types = _types_of(graph, r["o"], object_types_cache)
                    if obj_types:
                        pairs.update((r["p"], t) for t in obj_types)
                        continue
                # literal, or an untyped object IRI we know nothing more about
                pairs.add((r["p"], r["o"]))
            instance_pairs[inst] = pairs
            for pair in pairs:
                pair_counts[pair] += 1

        n = len(instances)
        expected = {pair for pair, c in pair_counts.items() if c / n >= threshold}
        if not expected:
            continue

        pair_missing_instances: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for inst, pairs in instance_pairs.items():
            if subject_filter and inst != subject_filter:
                continue
            for pair in expected - pairs:
                pair_missing_instances[pair].append(inst)

        for (p, o), missing_instances in pair_missing_instances.items():
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "object_type": o,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{pair_counts[(p, o)]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["object_type"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# E. Predicate-object gaps, INSTANCE level ("peers have this exact fact, some
#    of you don't") - per class, per (predicate, object), for all instances
#    or one subject
# --------------------------------------------------------------------------- #

def find_predicate_object_instances_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
) -> List[Dict]:
    """
    For every class (or just `class_filter` if given), find (predicate,
    object) combinations that a majority of instances of that class share,
    then, per (class, predicate, object), report every instance lacking that
    specific fact as its own row - even if it has some other value for the
    same predicate. Each row carries the group's
    `missing_count`/`total_instances`/`peer_coverage` for context, plus a
    `subject` field naming exactly which instance the row is about,
    `subject_triples`: every other (predicate, object) fact already known
    about that subject (rdf:type included), and `object_triples`: one more
    hop out - every triple whose subject is an IRI object from
    `subject_triples` that is itself an instance (has its own rdf:type).

    Object IRIs are compared by exact identity here (e.g. hasPatient=wine1 is
    a different fact from hasPatient=wine2), which means this tends to
    fragment into many small, individual-specific gaps whenever a predicate
    points at many distinct individuals of the same type. For the coarser,
    type-level version of this check (comparing "a Wine" rather than
    "wine1"), see find_predicate_object_gaps().

    Pairs and their peer counts are computed once per class and then reused
    for every instance in that class, rather than re-querying peers per
    subject.

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class (same as find_predicate_gaps()) - e.g.
    class_filter=".../Person" pulls in instances of ".../Student" etc. too.

    If `subject_filter` is given, only gaps touching that one subject are
    returned (but the majority is still computed over ALL peers in its
    class(es), consistent with the original single-subject behaviour) - in
    that case every returned row's "subject" will be that one subject.
    Blank-node objects are excluded since they have no stable identity to
    compare across instances; literals and IRIs are both compared as-is.
    """
    if class_filter:
        classes = [class_filter]
    else:
        classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})

    instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}
    object_types_cache: Dict[str, List[str]] = {}

    gaps = []
    for cls in classes:
        instances = [
            r["instance"]
            for r in run_query(graph, instances_query % (cls, max_instances_per_class))
        ]
        if len(instances) < 2:
            continue  # nothing to compare against

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        instance_pairs: Dict[str, Set[Tuple[str, str]]] = {}

        for inst in instances:
            pairs = {
                (r["p"], r["o"])
                for r in run_query(graph, PREDICATE_OBJECT_FOR_INSTANCE_QUERY % inst)
                if r["p"] != RDF_TYPE
            }
            instance_pairs[inst] = pairs
            for pair in pairs:
                pair_counts[pair] += 1

        n = len(instances)
        expected = {pair for pair, c in pair_counts.items() if c / n >= threshold}
        if not expected:
            continue

        pair_missing_instances: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for inst, pairs in instance_pairs.items():
            if subject_filter and inst != subject_filter:
                continue
            for pair in expected - pairs:
                pair_missing_instances[pair].append(inst)

        for (p, o), missing_instances in pair_missing_instances.items():
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "object": o,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{pair_counts[(p, o)]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["object"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# C. Dangling references (objects known only by name, nothing further known),
#    grouped by the (referencing type, predicate) that points at them
# --------------------------------------------------------------------------- #

DANGLING_QUERY = """
SELECT ?object ?p ?sType WHERE {
    ?s ?p ?object .
    OPTIONAL { ?s a ?sType }
    FILTER(isIRI(?object))
    FILTER NOT EXISTS { ?object ?p2 ?o2 . }
}
LIMIT %d
"""


def find_dangling_references(
    graph: Graph, row_limit: int = 5000, example_limit: int = DEFAULT_EXAMPLE_LIMIT
) -> List[Dict]:
    """
    Find objects that are referenced somewhere but have no outgoing triples
    of their own (we only know their name/id), grouped by the type of the
    referencing subject and the predicate used to reach them - e.g. "Person /
    hasFriend points at 12 references we know nothing further about". If the
    referencing subject itself has no declared type, it's grouped under
    "(untyped subject)".

    Note: since a dangling object has zero outgoing triples by definition,
    it cannot itself have an rdf:type - that's exactly the gap being
    reported. Grouping by the *referencing* type/predicate is what makes the
    finding pattern-level instead of a flat list of individual objects.
    """
    rows = run_query(graph, DANGLING_QUERY % row_limit)

    groups: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for r in rows:
        s_type = r["sType"] if r["sType"] is not None else "(untyped subject)"
        groups[(s_type, r["p"])].add(r["object"])

    results = []
    for (s_type, p), objs in groups.items():
        objs_sorted = sorted(objs)
        results.append(
            {
                "referencing_type": s_type,
                "predicate": p,
                "count": len(objs_sorted),
                "example_objects": objs_sorted[:example_limit],
            }
        )
    results.sort(key=lambda g: -g["count"])
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_report(
    graph: Graph,
    threshold: float,
    class_filter: str,
    subject_filter: str = None,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
) -> Dict:
    return {
        "untyped_entities": find_untyped_entities(graph, example_limit=example_limit),
        "predicate_gaps": find_predicate_gaps(
            graph, threshold=threshold, class_filter=class_filter, subject_filter=subject_filter
        ),
        "dangling_references": find_dangling_references(graph, example_limit=example_limit),
        "predicate_object_gaps": find_predicate_object_gaps(
            graph,
            threshold=threshold,
            class_filter=class_filter,
            subject_filter=subject_filter,
        ),
        "predicate_object_instances_gaps": find_predicate_object_instances_gaps(
            graph,
            threshold=threshold,
            class_filter=class_filter,
            subject_filter=subject_filter,
        ),
    }


def _print_subject_triples(subject_triples: List[Dict[str, str]]) -> None:
    """Print a subject's other known triples, indented under its gap row."""
    if not subject_triples:
        print("        (no other triples known for this subject)")
        return
    for t in subject_triples:
        print(f"        {t['predicate']} = {t['object']}")


def _print_object_triples(object_triples: List[Dict[str, str]]) -> None:
    """Print one-hop-out triples (facts about objects the subject points at), grouped by that object."""
    if not object_triples:
        return
    print("      object triples (facts about the subject's own instance-typed objects, one hop out):")
    last_subject = None
    for t in object_triples:
        if t["subject"] != last_subject:
            print(f"        {t['subject']}")
            last_subject = t["subject"]
        print(f"          {t['predicate']} = {t['object']}")


def print_report(report: Dict) -> None:
    print("\n=== A. Untyped entities, grouped by predicate signature ===")
    if not report["untyped_entities"]:
        print("  (none found)")
    for g in report["untyped_entities"][:50]:
        print(f"  - {g['instance_count']} untyped entities share predicates: {', '.join(g['predicate_signature']) or '(none)'}")
        for e in g["example_entities"]:
            print(f"      e.g. {e}")

    print("\n=== B. Predicate gaps (some instances missing what peers usually have) ===")
    if not report["predicate_gaps"]:
        print("  (none found)")
    for g in report["predicate_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_object_triples(g["object_triples"])

    print("\n=== C. Dangling references (named but nothing else known) ===")
    if not report["dangling_references"]:
        print("  (none found)")
    for g in report["dangling_references"][:50]:
        print(f"  - {g['referencing_type']}  --{g['predicate']}-->  {g['count']} dangling references")
        for o in g["example_objects"]:
            print(f"      e.g. {o}")

    print("\n=== D. Predicate-object gaps, type level (some instances missing a fact whose object TYPE most peers share) ===")
    if not report["predicate_object_gaps"]:
        print("  (none found)")
    for g in report["predicate_object_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']} -> (type) {g['object_type']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_object_triples(g["object_triples"])

    print("\n=== E. Predicate-object gaps, instance level (some instances missing the exact fact most peers share) ===")
    if not report["predicate_object_instances_gaps"]:
        print("  (none found)")
    for g in report["predicate_object_instances_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']} = {g['object']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_object_triples(g["object_triples"])

    # B, D, E are now one row per affected subject, so their row count IS the
    # affected-instance count; distinct (class, predicate[, object]) combos
    # are counted separately below for the "grouped into" summary.
    predicate_gap_groups = {(g["class"], g["predicate"]) for g in report["predicate_gaps"]}
    predicate_object_gap_groups = {
        (g["class"], g["predicate"], g["object_type"]) for g in report["predicate_object_gaps"]
    }
    predicate_object_instance_gap_groups = {
        (g["class"], g["predicate"], g["object"]) for g in report["predicate_object_instances_gaps"]
    }

    total = (
        sum(g["instance_count"] for g in report["untyped_entities"])
        + len(report["predicate_gaps"])
        + sum(g["count"] for g in report["dangling_references"])
        + len(report["predicate_object_gaps"])
        + len(report["predicate_object_instances_gaps"])
    )
    print(f"\nTotal affected instances/entities across all gap types: {total}")
    print(
        f"Grouped into {len(report['untyped_entities'])} untyped-signature group(s), "
        f"{len(predicate_gap_groups)} predicate-gap group(s) affecting {len(report['predicate_gaps'])} subject(s), "
        f"{len(report['dangling_references'])} dangling-reference group(s), "
        f"{len(predicate_object_gap_groups)} predicate-object-type-gap group(s) affecting {len(report['predicate_object_gaps'])} subject(s), "
        f"{len(predicate_object_instance_gap_groups)} predicate-object-instance-gap group(s) affecting {len(report['predicate_object_instances_gaps'])} subject(s)."
    )


def _write_json_report(report: Dict, json_path: str) -> None:
    """
    Write the report as JSON to `json_path`, which may be relative (resolved
    against the current working directory, same as plain open()) or
    absolute. Any missing parent directories are created first, so e.g.
    --json out/report.json works even if "out/" doesn't exist yet.
    """
    json_path = os.path.abspath(json_path)
    parent = os.path.dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)


# --------------------------------------------------------------------------- #
# Convenience entry point (e.g. for Colab / notebooks, no argparse needed)
# --------------------------------------------------------------------------- #

def analyze(
    file: str = None,
    endpoint: str = None,
    fmt: str = None,
    threshold: float = 0.6,
    class_uri: str = None,
    subject_uri: str = None,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    print_output: bool = True,
    json_path: str = None,
) -> Dict:
    """
    Convenience wrapper for calling the gap finder directly from a notebook
    (e.g. Colab) without going through the CLI/argparse.

    threshold is exposed here as a plain function parameter: the fraction of
    peers (0.0-1.0) that must share a predicate (or predicate-object pair)
    for it to count as "expected". Lower it (e.g. 0.5) to also catch 50/50
    splits like a 2-instance class where only one instance has a predicate.

    example_limit controls how many example entities are kept per group for
    drill-down in the untyped-entities (A) and dangling-references (C)
    sections (the reported counts always reflect the full group, regardless
    of this limit). The predicate-gap sections (B, D, E) instead report one
    row per affected subject, so example_limit does not apply to them.

    class_uri, if given, also covers (transitive) rdfs:subClassOf descendants
    of that class - e.g. class_uri=".../Person" also picks up instances of
    ".../Student" or ".../Employee".

    Example:
        report = analyze(file="mybrain.trig", fmt="trig", threshold=0.5)
    """
    if not file and not endpoint:
        raise ValueError("Provide either file= or endpoint=")

    if endpoint:
        graph = load_graph_from_endpoint(endpoint)
    else:
        graph = load_graph_from_file(file, fmt)

    report = build_report(
        graph,
        threshold=threshold,
        class_filter=class_uri,
        subject_filter=subject_uri,
        example_limit=example_limit,
    )

    if print_output:
        print_report(report)

    if json_path:
        _write_json_report(report, json_path)

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Query an RDF knowledge graph for knowledge gaps, grouped by type.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoint", help="SPARQL endpoint URL, e.g. http://localhost:7200/repositories/sandbox")
    src.add_argument("--file", help="Path to a local RDF file")
    parser.add_argument("--format", default=None, help="rdflib format for --file, e.g. turtle, trig, xml, json-ld")
    parser.add_argument("--threshold", type=float, default=0.6, help="Fraction of peers that must share a predicate (or predicate-object pair) for it to count as 'expected' (default 0.6)")
    parser.add_argument("--class-uri", default=None, help="Restrict predicate-gap / predicate-object-gap analysis to a single rdf:type URI (also includes its rdfs:subClassOf descendants)")
    parser.add_argument("--subject-uri", default=None, help="Restrict predicate gaps (sections B, D and E) to one specific subject URI instead of all instances")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLE_LIMIT, help=f"Number of example entities to keep per group for drill-down in sections A and C only (default {DEFAULT_EXAMPLE_LIMIT}); sections B, D, E always report one row per affected subject")
    parser.add_argument("--json", default=None, help="Optional path to write the full report as JSON; may be relative (resolved against the current working directory) or absolute, any missing parent directories are created automatically")
    args = parser.parse_args()

    if args.endpoint:
        graph = load_graph_from_endpoint(args.endpoint)
        print(f"Connected to SPARQL endpoint: {args.endpoint}")
    else:
        graph = load_graph_from_file(args.file, args.format)
        print(f"Loaded {len(graph)} triples from {args.file}")

    report = build_report(
        graph,
        threshold=args.threshold,
        class_filter=args.class_uri,
        subject_filter=args.subject_uri,
        example_limit=args.examples,
    )
    print_report(report)

    if args.json:
        _write_json_report(report, args.json)
        print(f"\nFull report written to {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
