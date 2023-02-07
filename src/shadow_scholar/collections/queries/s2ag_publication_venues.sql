UNLOAD
(SELECT
        id,
        alternate_issns,
        alternate_names,
        alternate_urls,
        issn,
        name,
        type,
        url
 FROM venues_prod_db.venues_exports_prod
 WHERE venues_prod_db.venues_exports_prod.year ={year}
 AND venues_prod_db.venues_exports_prod.month ={month}
 AND venues_prod_db.venues_exports_prod.day ={day}
 {limit}
)
TO '{output_location}'
WITH (format='json', compression='gzip')