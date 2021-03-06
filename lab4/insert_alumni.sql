PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <urn:absolute:/problem1a#>

INSERT {
    ?person rdfs:type :Person ;
              :personName ?personLabel ;
      		  :alumnusOf  ?univLabel  .
}
WHERE {
  SERVICE <https://query.wikidata.org/sparql>  {
    ?person wdt:P31 wd:Q5;
        wdt:P69 ?univ;
        rdfs:label ?personLabel.
      ?univ wdt:P31 wd:Q3918;
        wdt:P17 wd:Q34;
        rdfs:label ?univLabel.
      FILTER(LANGMATCHES(LANG(?personLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?univLabel), "sv"))
	}
}