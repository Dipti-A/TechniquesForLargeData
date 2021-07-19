PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>

PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <urn:absolute:/problem1a#>
INSERT {
    ?university rdf:type :Organisation ;
              :organisationName ?universityLabel .
}
WHERE {
  SERVICE <https://query.wikidata.org/sparql>  {
  ?university wdt:P31 wd:Q3918.
  ?university wdt:P17 wd:Q34.
  ?university rdfs:label ?universityLabel .
FILTER ( lang(?universityLabel) = "en" )
	}
}
