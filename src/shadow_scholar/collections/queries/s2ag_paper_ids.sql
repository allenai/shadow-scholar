UNLOAD
(SELECT paper_sha             as sha,
        corpus_paper_id       as corpusId,
        id_type = 'Canonical' as primary
 FROM content_ext.legacy_paper_ids
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')
