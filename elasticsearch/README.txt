* Checkout elasticsearch 7.8 revision 9c2c238baac2303c22632e0a9327a042ef0e9a82
* apply patch
* see start.sh to compile and start Elasticsearch with extra proximity measures

Example of query
POST  i5whole--gpt2-xl--w_250x64--tkn_same-sigmoid-text/_search
{
  "size": 100,
  "_source": "_id",
  "query": {
      "script_score": {
          "query": {
              "bool": {
                  "must_not": [
                      {"term": {"kicker": {"value": "opinion"}}},
                      {"term": {"kicker": {"value": "editorial"}}}
                  ]
              }
          },
          "script": {
              "source": "return 1/(1+customDistance(params.query_vector, 'embedding_state_last_hidden_mean', '003d:3'))",
              "params": {
                  "query_vector": [
  0.3897402330607126,
  0.4948820379478337,
  0.5235681362745803,
  0.7770907889820575,
...
  0.8884333628069284,
  0.4348203297547303,
  0.5202947642338975,
  0.5750202450755716
]
              }
          }
      }
  }
}

query_vector -- embedding vector of query article, first vector for distance
embedding_state_last_hidden_mean -- related documents vector, second vector for distance function
003d:3 -- proximity measure to use

List of available functions:
"001d" -- 1. Euclidean L2
"002d" -- 2. City block L1
"003d" -- 3. Minkowski Lp
"004d" -- 4. Chebyshev L∞
"005d" -- 5. Sørensen
"006d" -- 6. Gower already normalized
"007d":
"008d" -- 7. Soergel
"009d" -- 8. Kulczynski
"010d" -- 9. Canberra
"011d" -- 10. Lorentzian
"012s" -- 11. Intersection
"013d" -- 11. 1-012
"014d" -- 12. Wave Hedges
"015d" -- 12. Wave Hedges
"016s" -- 13. Czekanowski
"017d" -- 13. 1 - 016
"018s" -- 14. Motyka
"019d" -- 14. 1-018
"020s" -- 15. Kulczynski
"021s" -- 16. Ruzicka
"022d" -- 17. Tanimoto
"023d" -- 17. Tanimoto
"024s" -- 18. Inner Product
"025s" -- 19. Harmonic mean
"026s" -- 20. Cosine
"027s" -- 21. Kumar-Hassebrook (PCE)
"028s" -- 22. Jaccard
"029d" -- 22. Jaccard
"030s" -- 23. Dice
"031s" -- 23. Dice
"032s" -- 24. Fidelity
"033d" -- 25. Bhattacharyya
"034d" -- 26. Hellinger
"035d" -- 26. Hellinger
"036d" -- 26. Matusita
"037d" -- 26. Matusita
"038d" -- 28. Squared-chord
"039s" -- 28. Squared-chord
"040d" -- 29. Squared Euclidian
"041d" -- 30. Pearson χ²
"042d" -- 31. Neyman χ²
"043d" -- 32. Squared χ²
"044d" -- 33. Probabilistic Symmetric χ²
"045d" -- 34. Divergence
"046d" -- 35. Clark
"047d" -- 36. Additive Symmetric χ²
"048d" -- 37. Kullback-Leibler
"049d" -- 38. Jeffreys
"050d" -- 39. K divergence
"051d" -- 40. Topsøe
"052d" -- 41. Jensen-Shannon (same as 1/2 051d)
"053d" -- 42. Jensen difference
"054d" -- 43. Taneja
"055d" -- 44. Kumar-Johnson
"056d" -- 45. Avg (L1, L∞)
"060d" -- Vicis-Wave Hedges
"061d" -- Vicis-Symmetric χ²
"062d" -- Vicis-Symmetric χ²
"063d" -- Vicis-Symmetric χ²
"064d" -- max-Symmetric χ²
"065d" -- min-Symmetric χ²

Minkowski have extra parameter, for example "003d:3" is Minkowski L3, "003d:3" Minkowski L6
