UNLOAD
(SELECT corpusId
      , map(Array ['CorpusId','DOI','ArXiv','MAG','ACL','PubMed','PubMedCentral','DBLP'],
            Array [
                cast(corpusId as VARCHAR),
                doi.id,
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'ArXiv'), null,
                       (a, b) -> coalesce(a, b.id), x -> x),
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'MAG'), null,
                       (a, b) -> coalesce(a, b.id), x -> x),
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'ACL'), null,
                       (a, b) -> coalesce(a, b.id), x -> x),
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'Medline'), null,
                       (a, b) -> coalesce(a, regexp_replace(b.id, 'v\d+$', '')), x -> x),
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'PubMedCentral'), null,
                       (a, b) -> coalesce(a, b.id), x -> x),
                reduce(filter(sourceInfo.sourceIds, x -> x.source = 'DBLP'), null,
                       (a, b) -> coalesce(a, b.id), x -> x)
                ])                                                                        as externalIds
      , 'https://www.semanticscholar.org/paper/' || id                                    as url
      , title
      , transform(authors, a ->
                           map(Array ['name',
                                   'authorId'],
                               Array [a.name,
                                   CASE
                                       WHEN cardinality(coalesce(a.ids, ARRAY [])) > 0
                                           THEN cast(a.ids[1] as VARCHAR)
                                       ELSE null
                                       END]))                                             as authors
      , venue
      , venueId                                                                           as publicationVenueId
      , CASE
            WHEN pubDate IS NOT NULL THEN try_cast(substr(pubDate, 1, 4) as int)
            ELSE year END                                                                 as year
      , numCiting                                                                         as referenceCount
      , numCitedBy                                                                        as citationCount
      , numKeyCitations                                                                   as influentialCitationCount
      , coalesce(openAccessLocation.isPdf, false)                                         as isOpenAccess
      , transform(s2FieldsOfStudy,
                  x -> map(Array ['category','source'], Array [x, 's2-fos-model'])) ||
        transform(fieldsOfStudy, x ->
                                 map(Array ['category','source'], Array [x, 'external'])) as s2FieldsOfStudy
      , publicationTypes
      , pubDate                                                                           as publicationDate
      , journal
      , lastespressoupdate                                                                as updated
 FROM espresso.pq_paper
 WHERE partition_0 = '{export_date}' {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')

