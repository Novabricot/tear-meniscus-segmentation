## Rapport 2

### Etat de l'art 
Nous avons fini l'état de l'art de ce qui a été fait en IA appliquée aux yeux et notre maitre de stage nous a chacun demandé de choisir un dataset sur lequel nous aimerions travailler.
J'ai personnellement choisis un dataset répertoriant des images d'yeux avec la zone du ménisque lacrymal associée. J'ai choisis celui-ci car les données étaient propres et n'avaient pas d'application dans le milieu médical aujourd'hui. 

### Début du code
Avec ce dataset, j'ai d'abord essayer de recréer une baseline propre. Le papier de recherche associé à ce dataset mentionnait que leur meilleur résultat avait été avec U-Net et c'est donc ce sur quoi je me suis penchée. J'ai donc commencé par un rapide pré-processing afin de reconstruire la pipeline d'entraînement. 
Le F1-score atteint sur le papier était de 0.92 et sur le serveur de la fac, j'atteint un résultat de 0.91. J'ai ensuite voulu tester si une architecture plus récente et plus adaptée à la segmentation d'image marchait mieux. J'ai donc tester avec SegFormer et j'ai obtenu des résultats très similaires à ceux de U-Net. 
Mon maitre de stage m'avait lors d'une de nos première réunion suggéré d'utiliser une méthode de cross-validation puisque le dataset disposait de dossier dont les images avaient été prise dans des hôpitaux différents. J'ai donc implémenté une methode, leave-one-center-out. C'est à dire que chaque model était entrainé sur tous les centres sauf un sur lequel il s'entrainait. 
Les résultats n'ont pas été grandement améliorés, leur F1-Score variant entre 0.8763 et 0.8974 pour U-Net contre 0.8705 et 0.9036 pour SegFormer.

### Suite 
Suite à notre meeting avec les professeurs du laboratoire, nos encadrant étaient plutôt satisfait de mon avancé et validait les étapes que je prévoyait de faire. Je voudrais à présent refaire tourner l'entrainement, toujours en leave-one-center-out, mais en gardant cette fois uniquement entre image du même type (le dataset comportant des images RGB et Infrarouge) et tester des fundations modèles tels que SAM ou MedSAM