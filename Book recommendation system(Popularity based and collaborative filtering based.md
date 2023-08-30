# Book Recommender System (Popularity based and Collaborative Filtering Based)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


```python
books = pd.read_csv('books.csv',low_memory=False)

```


```python
ratings= pd.read_csv("ratings.csv")
```


```python
user= pd.read_csv("users.csv")
```


```python
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
      <th>Image-URL-S</th>
      <th>Image-URL-M</th>
      <th>Image-URL-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0195153448</td>
      <td>Classical Mythology</td>
      <td>Mark P. O. Morford</td>
      <td>2002</td>
      <td>Oxford University Press</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002005018</td>
      <td>Clara Callan</td>
      <td>Richard Bruce Wright</td>
      <td>2001</td>
      <td>HarperFlamingo Canada</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0060973129</td>
      <td>Decision in Normandy</td>
      <td>Carlo D'Este</td>
      <td>1991</td>
      <td>HarperPerennial</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0374157065</td>
      <td>Flu: The Story of the Great Influenza Pandemic...</td>
      <td>Gina Bari Kolata</td>
      <td>1999</td>
      <td>Farrar Straus Giroux</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0393045218</td>
      <td>The Mummies of Urumchi</td>
      <td>E. J. W. Barber</td>
      <td>1999</td>
      <td>W. W. Norton &amp;amp; Company</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>276725</td>
      <td>034545104X</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>276726</td>
      <td>0155061224</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>276727</td>
      <td>0446520802</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>276729</td>
      <td>052165615X</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>276729</td>
      <td>0521795028</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>Location</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>nyc, new york, usa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>stockton, california, usa</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>moscow, yukon territory, russia</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>porto, v.n.gaia, portugal</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>farnborough, hants, united kingdom</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Exploratory data analysis


```python
books.shape
```




    (271360, 8)




```python
ratings.shape
```




    (1149780, 3)




```python
users.shape
```




    (278858, 3)




```python
books.isnull().sum()
```




    ISBN                   0
    Book-Title             0
    Book-Author            2
    Year-Of-Publication    0
    Publisher              2
    Image-URL-S            0
    Image-URL-M            0
    Image-URL-L            3
    dtype: int64




```python
ratings.isnull().sum()
```




    User-ID        0
    ISBN           0
    Book-Rating    0
    dtype: int64




```python
users.isnull().sum()
```




    User-ID          0
    Location         0
    Age         110762
    dtype: int64




```python
books.duplicated().sum()
```




    0




```python
ratings.duplicated().sum()
```




    0




```python
users.duplicated().sum()
```




    0




```python
books["Publisher"].value_counts()
```




    Publisher
    Harlequin                  7535
    Silhouette                 4220
    Pocket                     3905
    Ballantine Books           3783
    Bantam Books               3646
                               ... 
    Hannover House                1
    Amber Quill Press, LLC.       1
    Lunchbox Press                1
    Ugly Town                     1
    Connaught                     1
    Name: count, Length: 16807, dtype: int64



## Popularity Based Recommender System


#### Top 50 books Highest Avergae rating but will consider only books with minimum of 250 votes

#### Merging Ratings Dataframe with Books


```python
ratings_with_name = ratings.merge(books,on='ISBN')
```


```python
ratings_with_name
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
      <th>Image-URL-S</th>
      <th>Image-URL-M</th>
      <th>Image-URL-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>276725</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2313</td>
      <td>034545104X</td>
      <td>5</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6543</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8680</td>
      <td>034545104X</td>
      <td>5</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10314</td>
      <td>034545104X</td>
      <td>9</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1031131</th>
      <td>276688</td>
      <td>0517145553</td>
      <td>0</td>
      <td>Mostly Harmless</td>
      <td>Douglas Adams</td>
      <td>1995</td>
      <td>Random House Value Pub</td>
      <td>http://images.amazon.com/images/P/0517145553.0...</td>
      <td>http://images.amazon.com/images/P/0517145553.0...</td>
      <td>http://images.amazon.com/images/P/0517145553.0...</td>
    </tr>
    <tr>
      <th>1031132</th>
      <td>276688</td>
      <td>1575660792</td>
      <td>7</td>
      <td>Gray Matter</td>
      <td>Shirley Kennett</td>
      <td>1996</td>
      <td>Kensington Publishing Corporation</td>
      <td>http://images.amazon.com/images/P/1575660792.0...</td>
      <td>http://images.amazon.com/images/P/1575660792.0...</td>
      <td>http://images.amazon.com/images/P/1575660792.0...</td>
    </tr>
    <tr>
      <th>1031133</th>
      <td>276690</td>
      <td>0590907301</td>
      <td>0</td>
      <td>Triplet Trouble and the Class Trip (Triplet Tr...</td>
      <td>Debbie Dadey</td>
      <td>1997</td>
      <td>Apple</td>
      <td>http://images.amazon.com/images/P/0590907301.0...</td>
      <td>http://images.amazon.com/images/P/0590907301.0...</td>
      <td>http://images.amazon.com/images/P/0590907301.0...</td>
    </tr>
    <tr>
      <th>1031134</th>
      <td>276704</td>
      <td>0679752714</td>
      <td>0</td>
      <td>A Desert of Pure Feeling (Vintage Contemporaries)</td>
      <td>Judith Freeman</td>
      <td>1997</td>
      <td>Vintage Books USA</td>
      <td>http://images.amazon.com/images/P/0679752714.0...</td>
      <td>http://images.amazon.com/images/P/0679752714.0...</td>
      <td>http://images.amazon.com/images/P/0679752714.0...</td>
    </tr>
    <tr>
      <th>1031135</th>
      <td>276704</td>
      <td>0806917695</td>
      <td>5</td>
      <td>Perplexing Lateral Thinking Puzzles: Scholasti...</td>
      <td>Paul Sloane</td>
      <td>1997</td>
      <td>Sterling Publishing</td>
      <td>http://images.amazon.com/images/P/0806917695.0...</td>
      <td>http://images.amazon.com/images/P/0806917695.0...</td>
      <td>http://images.amazon.com/images/P/0806917695.0...</td>
    </tr>
  </tbody>
</table>
<p>1031136 rows × 10 columns</p>
</div>



#### Group By Book- title and Book-Rating


```python
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book-Title</th>
      <th>num_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ask Lily (Young Women of Faith: Lily Series, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>241066</th>
      <td>Ã?Â?lpiraten.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>241067</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>4</td>
    </tr>
    <tr>
      <th>241068</th>
      <td>Ã?Â?sterlich leben.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>241069</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>3</td>
    </tr>
    <tr>
      <th>241070</th>
      <td>Ã?Â?thique en toc</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>241071 rows × 2 columns</p>
</div>



### Calculating Average Rating


```python
avg_rating_df = ratings_with_name.groupby(['Book-Title'])['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book-Title</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>2.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ask Lily (Young Women of Faith: Lily Series, ...</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>241066</th>
      <td>Ã?Â?lpiraten.</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>241067</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>241068</th>
      <td>Ã?Â?sterlich leben.</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>241069</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>241070</th>
      <td>Ã?Â?thique en toc</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
<p>241071 rows × 2 columns</p>
</div>



### Merging both dataframes


```python
popular_df= num_rating_df.merge(avg_rating_df,on="Book-Title")
popular_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book-Title</th>
      <th>num_ratings</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>4</td>
      <td>2.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ask Lily (Young Women of Faith: Lily Series, ...</td>
      <td>1</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>241066</th>
      <td>Ã?Â?lpiraten.</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>241067</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>4</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>241068</th>
      <td>Ã?Â?sterlich leben.</td>
      <td>1</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>241069</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>3</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>241070</th>
      <td>Ã?Â?thique en toc</td>
      <td>2</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
<p>241071 rows × 3 columns</p>
</div>



### Keeping the books with num_rating greater or equal to 250


```python
popular_df[popular_df["num_ratings"]>= 250].sort_values('avg_rating', ascending=False).head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book-Title</th>
      <th>num_ratings</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80434</th>
      <td>Harry Potter and the Prisoner of Azkaban (Book 3)</td>
      <td>428</td>
      <td>5.852804</td>
    </tr>
    <tr>
      <th>80422</th>
      <td>Harry Potter and the Goblet of Fire (Book 4)</td>
      <td>387</td>
      <td>5.824289</td>
    </tr>
    <tr>
      <th>80441</th>
      <td>Harry Potter and the Sorcerer's Stone (Book 1)</td>
      <td>278</td>
      <td>5.737410</td>
    </tr>
    <tr>
      <th>80426</th>
      <td>Harry Potter and the Order of the Phoenix (Boo...</td>
      <td>347</td>
      <td>5.501441</td>
    </tr>
    <tr>
      <th>80414</th>
      <td>Harry Potter and the Chamber of Secrets (Book 2)</td>
      <td>556</td>
      <td>5.183453</td>
    </tr>
    <tr>
      <th>191612</th>
      <td>The Hobbit : The Enchanting Prelude to The Lor...</td>
      <td>281</td>
      <td>5.007117</td>
    </tr>
    <tr>
      <th>187377</th>
      <td>The Fellowship of the Ring (The Lord of the Ri...</td>
      <td>368</td>
      <td>4.948370</td>
    </tr>
    <tr>
      <th>80445</th>
      <td>Harry Potter and the Sorcerer's Stone (Harry P...</td>
      <td>575</td>
      <td>4.895652</td>
    </tr>
    <tr>
      <th>211384</th>
      <td>The Two Towers (The Lord of the Rings, Part 2)</td>
      <td>260</td>
      <td>4.880769</td>
    </tr>
    <tr>
      <th>219741</th>
      <td>To Kill a Mockingbird</td>
      <td>510</td>
      <td>4.700000</td>
    </tr>
    <tr>
      <th>183573</th>
      <td>The Da Vinci Code</td>
      <td>898</td>
      <td>4.642539</td>
    </tr>
    <tr>
      <th>187880</th>
      <td>The Five People You Meet in Heaven</td>
      <td>430</td>
      <td>4.551163</td>
    </tr>
    <tr>
      <th>180556</th>
      <td>The Catcher in the Rye</td>
      <td>449</td>
      <td>4.545657</td>
    </tr>
    <tr>
      <th>196326</th>
      <td>The Lovely Bones: A Novel</td>
      <td>1295</td>
      <td>4.468726</td>
    </tr>
    <tr>
      <th>764</th>
      <td>1984</td>
      <td>284</td>
      <td>4.454225</td>
    </tr>
    <tr>
      <th>144165</th>
      <td>Prodigal Summer: A Novel</td>
      <td>253</td>
      <td>4.450593</td>
    </tr>
    <tr>
      <th>128670</th>
      <td>Neverwhere</td>
      <td>265</td>
      <td>4.449057</td>
    </tr>
    <tr>
      <th>206502</th>
      <td>The Secret Life of Bees</td>
      <td>774</td>
      <td>4.447028</td>
    </tr>
    <tr>
      <th>168719</th>
      <td>Stupid White Men ...and Other Sorry Excuses fo...</td>
      <td>283</td>
      <td>4.356890</td>
    </tr>
    <tr>
      <th>223135</th>
      <td>Tuesdays with Morrie: An Old Man, a Young Man,...</td>
      <td>493</td>
      <td>4.354970</td>
    </tr>
    <tr>
      <th>204387</th>
      <td>The Red Tent (Bestselling Backlist)</td>
      <td>723</td>
      <td>4.334716</td>
    </tr>
    <tr>
      <th>191589</th>
      <td>The Hitchhiker's Guide to the Galaxy</td>
      <td>268</td>
      <td>4.328358</td>
    </tr>
    <tr>
      <th>129379</th>
      <td>Nickel and Dimed: On (Not) Getting By in America</td>
      <td>335</td>
      <td>4.289552</td>
    </tr>
    <tr>
      <th>93381</th>
      <td>Into the Wild</td>
      <td>252</td>
      <td>4.273810</td>
    </tr>
    <tr>
      <th>63867</th>
      <td>Fahrenheit 451</td>
      <td>409</td>
      <td>4.264059</td>
    </tr>
    <tr>
      <th>74750</th>
      <td>Girl with a Pearl Earring</td>
      <td>526</td>
      <td>4.218631</td>
    </tr>
    <tr>
      <th>136145</th>
      <td>Outlander</td>
      <td>283</td>
      <td>4.173145</td>
    </tr>
    <tr>
      <th>233370</th>
      <td>Where the Heart Is (Oprah's Book Club (Paperba...</td>
      <td>585</td>
      <td>4.105983</td>
    </tr>
    <tr>
      <th>156102</th>
      <td>Seabiscuit: An American Legend</td>
      <td>275</td>
      <td>4.098182</td>
    </tr>
    <tr>
      <th>107962</th>
      <td>Life of Pi</td>
      <td>664</td>
      <td>4.088855</td>
    </tr>
    <tr>
      <th>176845</th>
      <td>The Bean Trees</td>
      <td>389</td>
      <td>4.087404</td>
    </tr>
    <tr>
      <th>2281</th>
      <td>A Child Called \It\": One Child's Courage to S...</td>
      <td>265</td>
      <td>4.086792</td>
    </tr>
    <tr>
      <th>8434</th>
      <td>ANGELA'S ASHES</td>
      <td>279</td>
      <td>4.075269</td>
    </tr>
    <tr>
      <th>76343</th>
      <td>Good in Bed</td>
      <td>490</td>
      <td>4.055102</td>
    </tr>
    <tr>
      <th>64931</th>
      <td>Fast Food Nation: The Dark Side of the All-Ame...</td>
      <td>321</td>
      <td>4.037383</td>
    </tr>
    <tr>
      <th>12700</th>
      <td>American Gods</td>
      <td>302</td>
      <td>4.006623</td>
    </tr>
    <tr>
      <th>161645</th>
      <td>Skipping Christmas</td>
      <td>322</td>
      <td>4.006211</td>
    </tr>
    <tr>
      <th>105777</th>
      <td>Left Behind: A Novel of the Earth's Last Days ...</td>
      <td>318</td>
      <td>4.003145</td>
    </tr>
    <tr>
      <th>189551</th>
      <td>The Golden Compass (His Dark Materials, Book 1)</td>
      <td>336</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>181679</th>
      <td>The Color Purple</td>
      <td>314</td>
      <td>3.964968</td>
    </tr>
    <tr>
      <th>160336</th>
      <td>Silence of the Lambs</td>
      <td>256</td>
      <td>3.960938</td>
    </tr>
    <tr>
      <th>8752</th>
      <td>About a Boy</td>
      <td>262</td>
      <td>3.900763</td>
    </tr>
    <tr>
      <th>158138</th>
      <td>Seven Up (A Stephanie Plum Novel)</td>
      <td>278</td>
      <td>3.888489</td>
    </tr>
    <tr>
      <th>175097</th>
      <td>The Alchemist: A Fable About Following Your Dream</td>
      <td>266</td>
      <td>3.875940</td>
    </tr>
    <tr>
      <th>80069</th>
      <td>Hard Eight : A Stephanie Plum Novel (A Stephan...</td>
      <td>269</td>
      <td>3.825279</td>
    </tr>
    <tr>
      <th>170101</th>
      <td>Suzanne's Diary for Nicholas</td>
      <td>457</td>
      <td>3.820569</td>
    </tr>
    <tr>
      <th>111073</th>
      <td>Lord of the Flies</td>
      <td>259</td>
      <td>3.818533</td>
    </tr>
    <tr>
      <th>5664</th>
      <td>A Prayer for Owen Meany</td>
      <td>413</td>
      <td>3.796610</td>
    </tr>
    <tr>
      <th>212070</th>
      <td>The Vampire Lestat (Vampire Chronicles, Book II)</td>
      <td>301</td>
      <td>3.777409</td>
    </tr>
    <tr>
      <th>233851</th>
      <td>White Oleander : A Novel (Oprah's Book Club)</td>
      <td>356</td>
      <td>3.772472</td>
    </tr>
  </tbody>
</table>
</div>



### Merge popular dataframe with Books


```python
popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]
```


```python
popular_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Image-URL-M</th>
      <th>num_ratings</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>Karen Hesse</td>
      <td>http://images.amazon.com/images/P/0590567330.0...</td>
      <td>4</td>
      <td>2.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>Rebecca Harvin</td>
      <td>http://images.amazon.com/images/P/0964147726.0...</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>Martina Boudreau</td>
      <td>http://images.amazon.com/images/P/0942320093.0...</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ask Lily (Young Women of Faith: Lily Series, ...</td>
      <td>Nancy N. Rue</td>
      <td>http://images.amazon.com/images/P/0310232546.0...</td>
      <td>1</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>Lou Mobley</td>
      <td>http://images.amazon.com/images/P/0962295701.0...</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>270291</th>
      <td>Ã?Â?lpiraten.</td>
      <td>Janwillem van de Wetering</td>
      <td>http://images.amazon.com/images/P/3499232499.0...</td>
      <td>2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>270292</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>Joan Aiken</td>
      <td>http://images.amazon.com/images/P/325721538X.0...</td>
      <td>4</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>270293</th>
      <td>Ã?Â?sterlich leben.</td>
      <td>Anselm GrÃ?Â¼n</td>
      <td>http://images.amazon.com/images/P/3451274973.0...</td>
      <td>1</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>270294</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>David Guterson</td>
      <td>http://images.amazon.com/images/P/3442725739.0...</td>
      <td>3</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>270295</th>
      <td>Ã?Â?thique en toc</td>
      <td>Didier Daeninckx</td>
      <td>http://images.amazon.com/images/P/2842192508.0...</td>
      <td>2</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
<p>241071 rows × 5 columns</p>
</div>




```python
popular_df['Image-URL-M'][0]
```




    'http://images.amazon.com/images/P/0590567330.01.MZZZZZZZ.jpg'



## Collaborative Filtering Based Recommender System


```python
#### Group by User-ID with minimum 200 ratings and Book rating 
```


```python
x= ratings_with_name.groupby('User-ID').count()["Book-Rating"]>200
active_users=x[x].index
```


```python
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]
filtered_rating
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
      <th>Image-URL-S</th>
      <th>Image-URL-M</th>
      <th>Image-URL-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>6543</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23768</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>28523</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>77940</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>81977</td>
      <td>034545104X</td>
      <td>0</td>
      <td>Flesh Tones: A Novel</td>
      <td>M. J. Rose</td>
      <td>2002</td>
      <td>Ballantine Books</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
      <td>http://images.amazon.com/images/P/034545104X.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1030883</th>
      <td>275970</td>
      <td>1880837927</td>
      <td>0</td>
      <td>The Theology of the Hammer</td>
      <td>Millard Fuller</td>
      <td>1994</td>
      <td>Smyth &amp;amp; Helwys Publishing</td>
      <td>http://images.amazon.com/images/P/1880837927.0...</td>
      <td>http://images.amazon.com/images/P/1880837927.0...</td>
      <td>http://images.amazon.com/images/P/1880837927.0...</td>
    </tr>
    <tr>
      <th>1030884</th>
      <td>275970</td>
      <td>188717897X</td>
      <td>0</td>
      <td>The Ordeal of Integration: Progress and Resent...</td>
      <td>Orlando Patterson</td>
      <td>1998</td>
      <td>Civitas Book Publisher</td>
      <td>http://images.amazon.com/images/P/188717897X.0...</td>
      <td>http://images.amazon.com/images/P/188717897X.0...</td>
      <td>http://images.amazon.com/images/P/188717897X.0...</td>
    </tr>
    <tr>
      <th>1030885</th>
      <td>275970</td>
      <td>1888889047</td>
      <td>0</td>
      <td>Pushcart's Complete Rotten Reviews &amp;amp; Rejec...</td>
      <td>Bill Henderson</td>
      <td>1998</td>
      <td>Pushcart Press</td>
      <td>http://images.amazon.com/images/P/1888889047.0...</td>
      <td>http://images.amazon.com/images/P/1888889047.0...</td>
      <td>http://images.amazon.com/images/P/1888889047.0...</td>
    </tr>
    <tr>
      <th>1030886</th>
      <td>275970</td>
      <td>1931868123</td>
      <td>0</td>
      <td>There's a Porcupine in My Outhouse: Misadventu...</td>
      <td>Mike Tougias</td>
      <td>2002</td>
      <td>Capital Books (VA)</td>
      <td>http://images.amazon.com/images/P/1931868123.0...</td>
      <td>http://images.amazon.com/images/P/1931868123.0...</td>
      <td>http://images.amazon.com/images/P/1931868123.0...</td>
    </tr>
    <tr>
      <th>1030887</th>
      <td>275970</td>
      <td>3411086211</td>
      <td>10</td>
      <td>Die Biene.</td>
      <td>Sybil GrÃ?Â¤fin SchÃ?Â¶nfeldt</td>
      <td>1993</td>
      <td>Bibliographisches Institut, Mannheim</td>
      <td>http://images.amazon.com/images/P/3411086211.0...</td>
      <td>http://images.amazon.com/images/P/3411086211.0...</td>
      <td>http://images.amazon.com/images/P/3411086211.0...</td>
    </tr>
  </tbody>
</table>
<p>474007 rows × 10 columns</p>
</div>



#### Group by Book title with minimum 50 ratings and Book rating 


```python
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index

```


```python
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
```


```python
final_ratings
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>ISBN</th>
      <th>Book-Rating</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
      <th>Image-URL-S</th>
      <th>Image-URL-M</th>
      <th>Image-URL-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>278418</td>
      <td>0446520802</td>
      <td>0</td>
      <td>The Notebook</td>
      <td>Nicholas Sparks</td>
      <td>1996</td>
      <td>Warner Books</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
    </tr>
    <tr>
      <th>65</th>
      <td>3363</td>
      <td>0446520802</td>
      <td>0</td>
      <td>The Notebook</td>
      <td>Nicholas Sparks</td>
      <td>1996</td>
      <td>Warner Books</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>7158</td>
      <td>0446520802</td>
      <td>10</td>
      <td>The Notebook</td>
      <td>Nicholas Sparks</td>
      <td>1996</td>
      <td>Warner Books</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>11676</td>
      <td>0446520802</td>
      <td>10</td>
      <td>The Notebook</td>
      <td>Nicholas Sparks</td>
      <td>1996</td>
      <td>Warner Books</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>23768</td>
      <td>0446520802</td>
      <td>6</td>
      <td>The Notebook</td>
      <td>Nicholas Sparks</td>
      <td>1996</td>
      <td>Warner Books</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
      <td>http://images.amazon.com/images/P/0446520802.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1026724</th>
      <td>266865</td>
      <td>0531001725</td>
      <td>10</td>
      <td>The Catcher in the Rye</td>
      <td>Jerome David Salinger</td>
      <td>1973</td>
      <td>Scholastic Library Pub</td>
      <td>http://images.amazon.com/images/P/0531001725.0...</td>
      <td>http://images.amazon.com/images/P/0531001725.0...</td>
      <td>http://images.amazon.com/images/P/0531001725.0...</td>
    </tr>
    <tr>
      <th>1027923</th>
      <td>269566</td>
      <td>0670809381</td>
      <td>0</td>
      <td>Echoes</td>
      <td>Maeve Binchy</td>
      <td>1986</td>
      <td>Penguin USA</td>
      <td>http://images.amazon.com/images/P/0670809381.0...</td>
      <td>http://images.amazon.com/images/P/0670809381.0...</td>
      <td>http://images.amazon.com/images/P/0670809381.0...</td>
    </tr>
    <tr>
      <th>1028777</th>
      <td>271284</td>
      <td>0440910927</td>
      <td>0</td>
      <td>The Rainmaker</td>
      <td>John Grisham</td>
      <td>1995</td>
      <td>Island</td>
      <td>http://images.amazon.com/images/P/0440910927.0...</td>
      <td>http://images.amazon.com/images/P/0440910927.0...</td>
      <td>http://images.amazon.com/images/P/0440910927.0...</td>
    </tr>
    <tr>
      <th>1029070</th>
      <td>271705</td>
      <td>B0001PIOX4</td>
      <td>0</td>
      <td>Fahrenheit 451</td>
      <td>Ray Bradbury</td>
      <td>1993</td>
      <td>Simon &amp;amp; Schuster</td>
      <td>http://images.amazon.com/images/P/B0001PIOX4.0...</td>
      <td>http://images.amazon.com/images/P/B0001PIOX4.0...</td>
      <td>http://images.amazon.com/images/P/B0001PIOX4.0...</td>
    </tr>
    <tr>
      <th>1030868</th>
      <td>275970</td>
      <td>1586210661</td>
      <td>9</td>
      <td>Me Talk Pretty One Day</td>
      <td>David Sedaris</td>
      <td>2001</td>
      <td>Time Warner Audio Major</td>
      <td>http://images.amazon.com/images/P/1586210661.0...</td>
      <td>http://images.amazon.com/images/P/1586210661.0...</td>
      <td>http://images.amazon.com/images/P/1586210661.0...</td>
    </tr>
  </tbody>
</table>
<p>58586 rows × 10 columns</p>
</div>



### Use Pivot Table to make dataframe


```python
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
```


```python
pt.fillna(0,inplace=True)
```


```python
pt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>User-ID</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>4017</th>
      <th>4385</th>
      <th>6251</th>
      <th>6323</th>
      <th>6543</th>
      <th>...</th>
      <th>271705</th>
      <th>273979</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>275970</th>
      <th>277427</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>Book-Title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>A Bend in the Road</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Year of Wonders</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>You Belong To Me</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Zen and the Art of Motorcycle Maintenance: An Inquiry into Values</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Zoya</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>\O\" Is for Outlaw"</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>706 rows × 810 columns</p>
</div>




```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
similarity_scores = cosine_similarity(pt)
```


```python
similarity_scores.shape
```




    (706, 706)




```python
def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
   
    for i in similar_items:
        print(pt.index[i[0]])
```


```python
recommend('1984')
```

    Animal Farm
    The Handmaid's Tale
    Brave New World
    The Vampire Lestat (Vampire Chronicles, Book II)



```python

```
