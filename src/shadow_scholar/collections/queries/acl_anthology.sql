UNLOAD (
    WITH acl_papers AS (
        SELECT corpus_paper_id AS id
        FROM content_ext.paper_sources
        WHERE source = 'ACL'
    ),
    s2orc_subset AS (
        SELECT
            s2orc.*
        FROM "s2orc_papers"."oa_latest" AS s2orc
        INNER JOIN acl_papers AS acl
            ON s2orc.id = acl.id
    ),
    citations_table AS (
        SELECT
            cits.citing_corpus_paperid AS id,
            ARRAY_AGG(cits.cited_corpus_paperid) AS citations
        FROM "content_ext"."citations"  AS cits
        INNER JOIN acl_papers AS sub
            ON cits.citing_corpus_paperid = sub.id
        GROUP BY cits.citing_corpus_paperid
    ),
    tldr_table AS (
        SELECT
            corpus_id AS id,
            model_output AS tldr
        FROM tldrs.v2_0_0 AS tldr
        INNER JOIN acl_papers AS sub
            ON tldr.id = sub.id
    )
    SELECT
        s2orc.*,
        ct.citations,
        tldr.tldr
    FROM s2orc_subset AS s2orc
    INNER JOIN citations_table AS ct
        ON s2orc.id = ct.id
    INNER JOIN tldr_table AS tldr
        ON s2orc.id = tldr.id
)
TO '{s3_output_location}'
WITH (
    format='JSON',
    compression='GZIP'
)
