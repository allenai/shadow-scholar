UNLOAD
(SELECT id                                              as authorId
      , CASE
            WHEN orcidId IS NOT NULL or dblpId IS NOT NULL
                THEN map(
                    Array ['DBLP','ORCID'],
                    Array [
                        dblpId,
                        CASE WHEN orcidId IS NOT NULL THEN ARRAY [orcidId] ELSE null END
                        ])
            ELSE null END                               as externalIds
      , 'https://www.semanticscholar.org/author/' || id as url
      , name
      , aliases
      , affiliations
      , homepage
      , statistics.totalPaperCount                      as paperCount
      , statistics.totalCitationCount                   as citationCount
      , statistics.hIndex                               as hIndex
      , lastespressoupdate as updated

 FROM espresso.pq_author
 WHERE partition_0 = '{export_date}'
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')

