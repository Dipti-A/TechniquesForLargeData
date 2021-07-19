PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX : <urn:absolute:/problem1a#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>

INSERT {
    ?organisation rdf:type :Organisation ;
              :organisationName ?organisationLabel .
}
WHERE {

    SERVICE <https://query.wikidata.org/sparql> {
        ?person wdt:P31 wd:Q5;
        wdt:P69 ?univ;
        wdt:P19 ?birthPlace;
        wdt:P27 ?country;
        wdt:P108 ?organisation.

       ?univ wdt:P31 wd:Q3918;
       wdt:P17 wd:Q34 ;
       wdt:P571 ?founded .

       ?organisation wdt:P131 ?orgnPlace;
       wdt:P17 ?orgnCountry.

       ?univ rdfs:label ?univLabel .
       ?birthPlace rdfs:label ?birthPlaceLabel .
       ?country rdfs:label ?countryLabel .
       ?organisation rdfs:label ?organisationLabel .
       ?person rdfs:label ?personLabel .
       ?orgnPlace rdfs:label ?orgnPlaceLabel .
       ?orgnCountry rdfs:label ?orgnCountryLabel .


       FILTER (langMatches( lang(?countryLabel), "EN" ) )
       FILTER (langMatches( lang(?univLabel), "EN" ) )
       FILTER (langMatches( lang(?birthPlaceLabel), "EN" ) )
       FILTER (langMatches( lang(?organisationLabel), "EN" ) )
       FILTER (langMatches( lang(?orgnPlaceLabel), "EN" ) )
       FILTER (langMatches( lang(?orgnCountryLabel), "EN" ) )
    }
}
