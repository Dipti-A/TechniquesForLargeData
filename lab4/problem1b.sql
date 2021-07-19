PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <urn:absolute:/problem1a#>

INSERT {
    ?person rdfs:type :Person ;
              :personName ?personLabel ;
      		  :alumnusOf  ?univLabel ;
    		  :employeeOf ?organisation;
    		  :bornIn ?birthPlace .
    ?birthPlace rdfs:type :Place ;
            :countryName ?countryLabel ;
    		:placeName ?placeLabel .
    ?hq rdfs:type :Place ;
            :countryName ?country2Label ;
    		:placeName ?hqLabel .
    ?organisation rdfs:type :Organisation ;
             :organisationName ?organisationLabel ;
    		 :locatedIn ?hq . 
}
WHERE {
  SERVICE <https://query.wikidata.org/sparql>  {
      ?person wdt:P31 wd:Q5; 
        wdt:P69 ?univ; 
        rdfs:label ?personLabel; 
        wdt:P108  ?organisation ;
        wdt:P19 ?birthPlace . 
            
      ?univ wdt:P31 wd:Q3918; 
        wdt:P17 wd:Q34; 
        wdt:P571 ?founded; 
        rdfs:label ?univLabel.
	
      ?birthPlace wdt:P131 ?country. 
      ?country wdt:P31 wd:Q6256. 
        
      ?organisation wdt:P159 ?hq .
      ?hq wdt:P131 ?country2 .
        
      ?country2 rdfs:label ?country2Label .
      ?hq rdfs:label ?hqLabel .
       
      ?birthPlace rdfs:label ?placeLabel. 
      ?country rdfs:label ?countryLabel.  
      ?organisation rdfs:label ?organisationLabel.
        
      FILTER(LANGMATCHES(LANG(?personLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?univLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?organisationLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?placeLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?countryLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?hqLabel), "sv"))
      FILTER(LANGMATCHES(LANG(?country2Label), "sv"))
	}
}
