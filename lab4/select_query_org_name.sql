PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <urn:absolute:/problem1a#>

SELECT?n
WHERE {
    ?b rdf:type :Organisation .
    ?b :organisationName ?n .
}