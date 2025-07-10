# Investigation-Netflix
Explorer les données des films Netflix et effectuer une analyse exploratoire des données pour une société de production afin de découvrir des informations sur les films d'une décennie particulière.


**Netflix** ! Ce qui a commencé en 1997 comme un service de location de DVD est devenu aujourd’hui l’une des plus grandes entreprises du divertissement et des médias.

Étant donné le grand nombre de films et de séries disponibles sur la plateforme, c’est l’occasion idéale de mettre en pratique tes compétences en analyse exploratoire de données et de plonger dans l’univers du divertissement.

Je suis amener à travailler pour une société de production spécialisée dans les styles nostalgiques. Je souhaite mener des recherches sur les films sortis dans les années 1990. Pour cela, je fais une exploration les données Netflix pour mieux comprendre cette formidable décennie cinématographique !

Le dataset `netflix_data.csv` a été fourni, accompagné du tableau ci-dessous qui détaille les noms et descriptions des colonnes. 

---

## Les données  
### **netflix_data.csv**

| Column | Description |
|--------|-------------|
| `show_id` | L’ID du programme |
| `type` | Type de programme |
| `title` | Titre du programme |
| `director` | Réalisateur du programme |
| `cast` | Distribution du programme |
| `country` | Pays d’origine |
| `date_added` | Date d’ajout sur Netflix |
| `release_year` | Année de sortie sur Netflix |
| `duration` | Durée du programme en minutes |
| `description` | Description du programme |
| `genre` | Genre du programme |

---

<center><img src="redpopcorn.jpg"></center>

 **Problématique : Effectuer une analyse exploratoire des données ``netflix_data.csv`` pour mieux comprendre les films de la décennie 1990**.

- Quelle était la durée la plus fréquente des films dans les années 1990 ? Enregistrer une réponse approximative sous forme d'entier duration(utilisez 1990 comme année de début de la décennie).

- Un film est considéré comme court s'il dure moins de 90 minutes. Comptez le nombre de courts métrages d'action sortis dans les années 1990 et enregistrez cet entier sous la forme short_movie_count.
