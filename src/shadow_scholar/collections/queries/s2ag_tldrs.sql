UNLOAD
(SELECT
        corpus_id as corpusId,
        'tldr@v2.0.0' as model,
        model_output as text

 FROM tldrs.v2_0_0
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')

