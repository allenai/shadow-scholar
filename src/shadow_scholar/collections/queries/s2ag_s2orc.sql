/*
 Some internal fields are dropped from the s2orc_papers.releases table, e.g. S3 location of the PDF.
 This is accomplished somewhat awkwardly by nesting row() structures and casting the whole thing
 into the desired output structure
 */

UNLOAD
(
    SELECT id                                         as corpusId,
           metadata.external_ids                      as externalIds,
           cast(
                   row (
                       row (
                           cast(json_parse(content.source.pdf_uri) as array(varchar)),
                           content.source.pdf_hash,
                           content.source.oa_info
                           ),
                       content.grobid.contents,
                       row (
                           content.grobid.annotations.abstract,
                           content.grobid.annotations.author,
                           content.grobid.annotations.author_affiliation,
                           content.grobid.annotations.author_first_name,
                           content.grobid.annotations.author_last_name,
                           content.grobid.annotations.bib_author,
                           content.grobid.annotations.bib_author_first_name,
                           content.grobid.annotations.bib_author_last_name,
                           content.grobid.annotations.bib_entry,
                           content.grobid.annotations.bib_ref,
                           content.grobid.annotations.bib_title,
                           content.grobid.annotations.bib_venue,
                           content.grobid.annotations.figure,
                           content.grobid.annotations.figure_caption,
                           content.grobid.annotations.figure_ref,
                           content.grobid.annotations.formula,
                           content.grobid.annotations.paragraph,
                           content.grobid.annotations.publisher,
                           content.grobid.annotations.section_header,
                           content.grobid.annotations."table",
                           content.grobid.annotations.table_ref,
                           content.grobid.annotations.title,
                           content.grobid.annotations.venue
                           )
                       )
               as row(source row(
                                 pdfUrls array(varchar),
                                 pdfSha varchar,
                                 oaInfo row(
                                             license varchar,
                                             openAccessUrl varchar,
                                             status varchar
                                             )
                                 ),
                      text varchar,
                      annotations row(
                                      abstract varchar,
                                      author varchar,
                                      authorAffiliation varchar,
                                      authorFirstName varchar,
                                      authorLastName varchar,
                                      bibAuthor varchar,
                                      bibAuthorFirstName varchar,
                                      bibAuthorLastName varchar,
                                      bibEntry varchar,
                                      bibRef varchar,
                                      bibTitle varchar,
                                      bibVenue varchar,
                                      figure varchar,
                                      figureCaption varchar,
                                      figureRef varchar,
                                      formula varchar,
                                      paragraph varchar,
                                      publisher varchar,
                                      sectionHeader varchar,
                                      "table" varchar,
                                      tableRef varchar,
                                      title varchar,
                                      venue varchar
                                      )
                      )
               )                                      as content,
           date_format(updated, '%Y-%m-%dT%H:%i:%sZ') as updated
    FROM s2orc_papers.oa_releases
    WHERE year ={year}
      and month ={month}
      and day ={day}
        {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')
