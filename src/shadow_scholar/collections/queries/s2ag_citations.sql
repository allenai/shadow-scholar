UNLOAD
(SELECT citingPaper.corpusId                                                          as citingCorpusId
      , citedPaper.corpusId                                                           as citedCorpusId
      , isKeyCitation                                                                 as isInfluential
      , nullif(transform(context, c -> c.string), ARRAY [])                           as contexts
      , nullif(array_distinct(flatten(transform(context, c -> c.intents))), ARRAY []) as intents
      , lastespressoupdate as updated
 FROM espresso.pq_citation
 WHERE partition_0 = '{export_date}'
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')

