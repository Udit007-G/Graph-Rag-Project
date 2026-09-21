GSCHEMA = """
CREATE GRAPH olympics (
    VERTEX Document (
        doc_id STRING PRIMARY KEY,
        title STRING,
        url STRING,
        wikidata_qid STRING,
        wikipedia_pageid INT64,
        approx_tokens INT64,
        text STRING
    )
    VERTEX Event (
        event_id STRING PRIMARY KEY,
        name STRING,
        sport STRING,
        year INT64,
        season STRING,
        venue STRING,
        date_str STRING
    )
    VERTEX Person (
        person_id STRING PRIMARY KEY,
        name STRING,
        nationality STRING,
        noc_code STRING
    )
    VERTEX Nation (
        noc_code STRING PRIMARY KEY,
        name STRING
    )
    VERTEX Medal (
        medal_id STRING PRIMARY KEY,
        event_id STRING,
        person_id STRING,
        medal_type STRING,
        year INT64
    )

    EDGE participated_in (
        FROM Person TO Event,
        STRING nation
    )
    EDGE won_medal (
        FROM Person TO Event,
        STRING medal_type,
        STRING nation
    )
    EDGE held_at (
        FROM Event TO Nation
    )
    EDGE belongs_to (
        FROM Person TO Nation
    )
    EDGE doc_mentions_event (
        FROM Document TO Event
    )
    EDGE doc_mentions_person (
        FROM Document TO Person
    )
    EDGE related_doc (
        FROM Document TO Document,
        STRING relation_type
    )
) WITH (
    STORE_TYPE = "hypergraph",
    DIRECTED = true
)
"""
