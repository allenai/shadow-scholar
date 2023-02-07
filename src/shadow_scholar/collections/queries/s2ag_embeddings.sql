UNLOAD
(SELECT
        paper_id as corpusId,
        'specter@v0.1.1' as model,
        embedding as vector,
        CASE WHEN updated > 0 THEN date_format(from_unixtime(updated),'%Y-%m-%dT%H:%i:%sZ') ELSE null END as updated

 FROM paper_embeddings.paper_embeddings_prod
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')

