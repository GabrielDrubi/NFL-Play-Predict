Skip to content
Search or jump to…
Pull requests
Issues
Codespaces
Marketplace
Explore
 
@GabrielDrubi 
GabrielDrubi
/
NFL-Play-Predict
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Delete ml.ipynb
 main
@GabrielDrubi
GabrielDrubi committed 12 minutes ago 
1 parent de068c0 commit b675b39a7edbedbfdb6c6874e908d21ed205c024
Showing 1 changed file with 0 additions and 1,642 deletions.
 1,642  
ml.ipynb
@@ -1,1642 +0,0 @@
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>play_id</th>\n",
       "      <th>game_id</th>\n",
       "      <th>home_team</th>\n",
       "      <th>away_team</th>\n",
       "      <th>posteam</th>\n",
       "      <th>posteam_type</th>\n",
       "      <th>defteam</th>\n",
       "      <th>side_of_field</th>\n",
       "      <th>yardline_100</th>\n",
       "      <th>game_date</th>\n",
       "      <th>...</th>\n",
       "      <th>penalty_player_id</th>\n",
       "      <th>penalty_player_name</th>\n",
       "      <th>penalty_yards</th>\n",
       "      <th>replay_or_challenge</th>\n",
       "      <th>replay_or_challenge_result</th>\n",
       "      <th>penalty_type</th>\n",
       "      <th>defensive_two_point_attempt</th>\n",
       "      <th>defensive_two_point_conv</th>\n",
       "      <th>defensive_extra_point_attempt</th>\n",
       "      <th>defensive_extra_point_conv</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>46</td>\n",
       "      <td>2009091000</td>\n",
       "      <td>PIT</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>TEN</td>\n",
       "      <td>TEN</td>\n",
       "      <td>30.0</td>\n",
       "      <td>2009-09-10</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>68</td>\n",
       "      <td>2009091000</td>\n",
       "      <td>PIT</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>58.0</td>\n",
       "      <td>2009-09-10</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>92</td>\n",
       "      <td>2009091000</td>\n",
       "      <td>PIT</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>53.0</td>\n",
       "      <td>2009-09-10</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>113</td>\n",
       "      <td>2009091000</td>\n",
       "      <td>PIT</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>56.0</td>\n",
       "      <td>2009-09-10</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>139</td>\n",
       "      <td>2009091000</td>\n",
       "      <td>PIT</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>TEN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>56.0</td>\n",
       "      <td>2009-09-10</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 255 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   play_id     game_id home_team away_team posteam posteam_type defteam  \\\n",
       "0       46  2009091000       PIT       TEN     PIT         home     TEN   \n",
       "1       68  2009091000       PIT       TEN     PIT         home     TEN   \n",
       "2       92  2009091000       PIT       TEN     PIT         home     TEN   \n",
       "3      113  2009091000       PIT       TEN     PIT         home     TEN   \n",
       "4      139  2009091000       PIT       TEN     PIT         home     TEN   \n",
       "\n",
       "  side_of_field  yardline_100   game_date  ...  penalty_player_id  \\\n",
       "0           TEN          30.0  2009-09-10  ...                NaN   \n",
       "1           PIT          58.0  2009-09-10  ...                NaN   \n",
       "2           PIT          53.0  2009-09-10  ...                NaN   \n",
       "3           PIT          56.0  2009-09-10  ...                NaN   \n",
       "4           PIT          56.0  2009-09-10  ...                NaN   \n",
       "\n",
       "   penalty_player_name  penalty_yards replay_or_challenge  \\\n",
       "0                  NaN            NaN                   0   \n",
       "1                  NaN            NaN                   0   \n",
       "2                  NaN            NaN                   0   \n",
       "3                  NaN            NaN                   0   \n",
       "4                  NaN            NaN                   0   \n",
       "\n",
       "   replay_or_challenge_result  penalty_type  defensive_two_point_attempt  \\\n",
       "0                         NaN           NaN                          0.0   \n",
       "1                         NaN           NaN                          0.0   \n",
       "2                         NaN           NaN                          0.0   \n",
       "3                         NaN           NaN                          0.0   \n",
       "4                         NaN           NaN                          0.0   \n",
       "\n",
       "   defensive_two_point_conv  defensive_extra_point_attempt  \\\n",
       "0                       0.0                            0.0   \n",
       "1                       0.0                            0.0   \n",
       "2                       0.0                            0.0   \n",
       "3                       0.0                            0.0   \n",
       "4                       0.0                            0.0   \n",
       "\n",
       "   defensive_extra_point_conv  \n",
       "0                         0.0  \n",
       "1                         0.0  \n",
       "2                         0.0  \n",
       "3                         0.0  \n",
       "4                         0.0  \n",
       "\n",
       "[5 rows x 255 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pickle\n",
    "import warnings \n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler\n",
    "\n",
    "#divisao treino e teste\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "#modelos\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn import linear_model\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "#metricas\n",
    "from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, plot_confusion_matrix \n",
    "\n",
    "#validacao\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "df = pd.read_csv(\"NFL Play by Play 2009-2018 (v5).csv\")\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Criando novas colunas que serao utilizadas no machine learning (Rushing Mean e Passing Mean)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>game_id</th>\n",
       "      <th>qtr</th>\n",
       "      <th>posteam</th>\n",
       "      <th>RushingMean</th>\n",
       "      <th>PassingMean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2009091000</td>\n",
       "      <td>1</td>\n",
       "      <td>PIT</td>\n",
       "      <td>0.333333</td>\n",
       "      <td>-0.125000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2009091000</td>\n",
       "      <td>1</td>\n",
       "      <td>TEN</td>\n",
       "      <td>6.166667</td>\n",
       "      <td>6.222222</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2009091000</td>\n",
       "      <td>2</td>\n",
       "      <td>PIT</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>9.062500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2009091000</td>\n",
       "      <td>2</td>\n",
       "      <td>TEN</td>\n",
       "      <td>1.400000</td>\n",
       "      <td>8.307692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2009091000</td>\n",
       "      <td>3</td>\n",
       "      <td>PIT</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      game_id  qtr posteam  RushingMean  PassingMean\n",
       "0  2009091000    1     PIT     0.333333    -0.125000\n",
       "1  2009091000    1     TEN     6.166667     6.222222\n",
       "2  2009091000    2     PIT     2.500000     9.062500\n",
       "3  2009091000    2     TEN     1.400000     8.307692\n",
       "4  2009091000    3     PIT     2.000000     3.000000"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r_off_agg = df[(df.play_type == 'run')]\n",
    "p_off_agg = df[(df.play_type == 'pass')|(df.play_type == 'sack')]\n",
    "\n",
    "r_off_agg = r_off_agg.groupby(['game_id','qtr','posteam'])['yards_gained'].mean().reset_index()\n",
    "p_off_agg = p_off_agg.groupby(['game_id','qtr','posteam'])['yards_gained'].mean().reset_index()\n",
    "\n",
    "r_off_agg = r_off_agg.rename(columns={'yards_gained':'RushingMean'})\n",
    "p_off_agg = p_off_agg.rename(columns={'yards_gained':'PassingMean'})\n",
    "\n",
    "off_agg = pd.merge(r_off_agg, p_off_agg, left_on=['game_id','qtr','posteam'], right_on=['game_id','qtr','posteam'], how='outer')\n",
    "\n",
    "off_agg.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.merge(off_agg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 436122 entries, 0 to 436121\n",
      "Columns: 257 entries, play_id to PassingMean\n",
      "dtypes: float64(137), int64(18), object(102)\n",
      "memory usage: 858.5+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Escolhendo as colunas que serao utilizadas no machine learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>qtr</th>\n",
       "      <th>yardline_100</th>\n",
       "      <th>ydstogo</th>\n",
       "      <th>half_seconds_remaining</th>\n",
       "      <th>down</th>\n",
       "      <th>score_differential</th>\n",
       "      <th>posteam</th>\n",
       "      <th>posteam_type</th>\n",
       "      <th>play_type</th>\n",
       "      <th>game_seconds_remaining</th>\n",
       "      <th>posteam_timeouts_remaining</th>\n",
       "      <th>PassingMean</th>\n",
       "      <th>RushingMean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1800.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>kickoff</td>\n",
       "      <td>3600.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>58.0</td>\n",
       "      <td>10</td>\n",
       "      <td>1793.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3593.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>53.0</td>\n",
       "      <td>5</td>\n",
       "      <td>1756.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>run</td>\n",
       "      <td>3556.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>56.0</td>\n",
       "      <td>8</td>\n",
       "      <td>1715.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3515.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>56.0</td>\n",
       "      <td>8</td>\n",
       "      <td>1707.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>punt</td>\n",
       "      <td>3507.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   qtr  yardline_100  ydstogo  half_seconds_remaining  down  \\\n",
       "0    1          30.0        0                  1800.0   NaN   \n",
       "1    1          58.0       10                  1793.0   1.0   \n",
       "2    1          53.0        5                  1756.0   2.0   \n",
       "3    1          56.0        8                  1715.0   3.0   \n",
       "4    1          56.0        8                  1707.0   4.0   \n",
       "\n",
       "   score_differential posteam posteam_type play_type  game_seconds_remaining  \\\n",
       "0                 NaN     PIT         home   kickoff                  3600.0   \n",
       "1                 0.0     PIT         home      pass                  3593.0   \n",
       "2                 0.0     PIT         home       run                  3556.0   \n",
       "3                 0.0     PIT         home      pass                  3515.0   \n",
       "4                 0.0     PIT         home      punt                  3507.0   \n",
       "\n",
       "   posteam_timeouts_remaining  PassingMean  RushingMean  \n",
       "0                         3.0       -0.125     0.333333  \n",
       "1                         3.0       -0.125     0.333333  \n",
       "2                         3.0       -0.125     0.333333  \n",
       "3                         3.0       -0.125     0.333333  \n",
       "4                         3.0       -0.125     0.333333  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df[[\"qtr\", \"yardline_100\", \"ydstogo\", \"half_seconds_remaining\",\"down\",\"score_differential\",\"posteam\",\"posteam_type\",\"play_type\",\"game_seconds_remaining\",\"posteam_timeouts_remaining\",\"PassingMean\",\"RushingMean\"]]\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Apagando as linhas que contem informacoes em branco"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "qtr                           0\n",
       "yardline_100                  0\n",
       "ydstogo                       0\n",
       "half_seconds_remaining        0\n",
       "down                          0\n",
       "score_differential            0\n",
       "posteam                       0\n",
       "posteam_type                  0\n",
       "play_type                     0\n",
       "game_seconds_remaining        0\n",
       "posteam_timeouts_remaining    0\n",
       "PassingMean                   0\n",
       "RushingMean                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.dropna()\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "qtr                              5\n",
       "yardline_100                    99\n",
       "ydstogo                         47\n",
       "half_seconds_remaining        1801\n",
       "down                             4\n",
       "score_differential             109\n",
       "posteam                         35\n",
       "posteam_type                     2\n",
       "play_type                        7\n",
       "game_seconds_remaining        3601\n",
       "posteam_timeouts_remaining       4\n",
       "PassingMean                   1780\n",
       "RushingMean                    877\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Apagando as linhas da coluna playtype que nao forem pass ou run"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def shorten_categories(categories, cutoff):\n",
    "    categorical_map = {}\n",
    "    for i in range(len(categories)):\n",
    "        if categories.values[i] >= cutoff:\n",
    "            categorical_map[categories.index[i]] = categories.index[i]\n",
    "        else:\n",
    "            categorical_map[categories.index[i]] = 'NaN'\n",
    "    return categorical_map"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pass    183949\n",
       "run     131843\n",
       "NaN      61650\n",
       "Name: play_type, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "country_map = shorten_categories(df.play_type.value_counts(), 100000)\n",
    "df['play_type'] = df['play_type'].map(country_map)\n",
    "df.play_type.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df =  df[df.play_type != 'NaN']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "qtr                              5\n",
       "yardline_100                    99\n",
       "ydstogo                         46\n",
       "half_seconds_remaining        1801\n",
       "down                             4\n",
       "score_differential             108\n",
       "posteam                         35\n",
       "posteam_type                     2\n",
       "play_type                        2\n",
       "game_seconds_remaining        3601\n",
       "posteam_timeouts_remaining       4\n",
       "PassingMean                   1780\n",
       "RushingMean                    877\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 315792 entries, 1 to 436121\n",
      "Data columns (total 13 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   qtr                         315792 non-null  int64  \n",
      " 1   yardline_100                315792 non-null  float64\n",
      " 2   ydstogo                     315792 non-null  int64  \n",
      " 3   half_seconds_remaining      315792 non-null  float64\n",
      " 4   down                        315792 non-null  float64\n",
      " 5   score_differential          315792 non-null  float64\n",
      " 6   posteam                     315792 non-null  object \n",
      " 7   posteam_type                315792 non-null  object \n",
      " 8   play_type                   315792 non-null  object \n",
      " 9   game_seconds_remaining      315792 non-null  float64\n",
      " 10  posteam_timeouts_remaining  315792 non-null  float64\n",
      " 11  PassingMean                 315792 non-null  float64\n",
      " 12  RushingMean                 315792 non-null  float64\n",
      "dtypes: float64(8), int64(2), object(3)\n",
      "memory usage: 33.7+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "qtr                           0\n",
       "yardline_100                  0\n",
       "ydstogo                       0\n",
       "half_seconds_remaining        0\n",
       "down                          0\n",
       "score_differential            0\n",
       "posteam                       0\n",
       "posteam_type                  0\n",
       "play_type                     0\n",
       "game_seconds_remaining        0\n",
       "posteam_timeouts_remaining    0\n",
       "PassingMean                   0\n",
       "RushingMean                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.dropna()\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 315792 entries, 1 to 436121\n",
      "Data columns (total 13 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   qtr                         315792 non-null  int64  \n",
      " 1   yardline_100                315792 non-null  float64\n",
      " 2   ydstogo                     315792 non-null  int64  \n",
      " 3   half_seconds_remaining      315792 non-null  float64\n",
      " 4   down                        315792 non-null  float64\n",
      " 5   score_differential          315792 non-null  float64\n",
      " 6   posteam                     315792 non-null  object \n",
      " 7   posteam_type                315792 non-null  object \n",
      " 8   play_type                   315792 non-null  object \n",
      " 9   game_seconds_remaining      315792 non-null  float64\n",
      " 10  posteam_timeouts_remaining  315792 non-null  float64\n",
      " 11  PassingMean                 315792 non-null  float64\n",
      " 12  RushingMean                 315792 non-null  float64\n",
      "dtypes: float64(8), int64(2), object(3)\n",
      "memory usage: 33.7+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>qtr</th>\n",
       "      <th>yardline_100</th>\n",
       "      <th>ydstogo</th>\n",
       "      <th>half_seconds_remaining</th>\n",
       "      <th>down</th>\n",
       "      <th>score_differential</th>\n",
       "      <th>posteam</th>\n",
       "      <th>posteam_type</th>\n",
       "      <th>play_type</th>\n",
       "      <th>game_seconds_remaining</th>\n",
       "      <th>posteam_timeouts_remaining</th>\n",
       "      <th>PassingMean</th>\n",
       "      <th>RushingMean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>58.0</td>\n",
       "      <td>10</td>\n",
       "      <td>1793.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3593.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>53.0</td>\n",
       "      <td>5</td>\n",
       "      <td>1756.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>run</td>\n",
       "      <td>3556.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>56.0</td>\n",
       "      <td>8</td>\n",
       "      <td>1715.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3515.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1</td>\n",
       "      <td>43.0</td>\n",
       "      <td>10</td>\n",
       "      <td>1584.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3384.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>40.0</td>\n",
       "      <td>7</td>\n",
       "      <td>1548.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>PIT</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>3348.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.125</td>\n",
       "      <td>0.333333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   qtr  yardline_100  ydstogo  half_seconds_remaining  down  \\\n",
       "1    1          58.0       10                  1793.0   1.0   \n",
       "2    1          53.0        5                  1756.0   2.0   \n",
       "3    1          56.0        8                  1715.0   3.0   \n",
       "5    1          43.0       10                  1584.0   1.0   \n",
       "6    1          40.0        7                  1548.0   2.0   \n",
       "\n",
       "   score_differential posteam posteam_type play_type  game_seconds_remaining  \\\n",
       "1                 0.0     PIT         home      pass                  3593.0   \n",
       "2                 0.0     PIT         home       run                  3556.0   \n",
       "3                 0.0     PIT         home      pass                  3515.0   \n",
       "5                 0.0     PIT         home      pass                  3384.0   \n",
       "6                 0.0     PIT         home      pass                  3348.0   \n",
       "\n",
       "   posteam_timeouts_remaining  PassingMean  RushingMean  \n",
       "1                         3.0       -0.125     0.333333  \n",
       "2                         3.0       -0.125     0.333333  \n",
       "3                         3.0       -0.125     0.333333  \n",
       "5                         3.0       -0.125     0.333333  \n",
       "6                         3.0       -0.125     0.333333  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Apagando 200.000 linhas porque estava tendo problemas ao fazer o machine learning devido ao tamanho."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "n = 200000\n",
    "df.drop(index=df.index[:n],inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Transformando colunas que possuem informacoes no formato string em int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 115792 entries, 276133 to 436121\n",
      "Data columns (total 13 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   qtr                         115792 non-null  int64  \n",
      " 1   yardline_100                115792 non-null  float64\n",
      " 2   ydstogo                     115792 non-null  int64  \n",
      " 3   half_seconds_remaining      115792 non-null  float64\n",
      " 4   down                        115792 non-null  float64\n",
      " 5   score_differential          115792 non-null  float64\n",
      " 6   posteam                     115792 non-null  object \n",
      " 7   posteam_type                115792 non-null  object \n",
      " 8   play_type                   115792 non-null  object \n",
      " 9   game_seconds_remaining      115792 non-null  float64\n",
      " 10  posteam_timeouts_remaining  115792 non-null  float64\n",
      " 11  PassingMean                 115792 non-null  float64\n",
      " 12  RushingMean                 115792 non-null  float64\n",
      "dtypes: float64(8), int64(2), object(3)\n",
      "memory usage: 12.4+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>qtr</th>\n",
       "      <th>yardline_100</th>\n",
       "      <th>ydstogo</th>\n",
       "      <th>half_seconds_remaining</th>\n",
       "      <th>down</th>\n",
       "      <th>score_differential</th>\n",
       "      <th>posteam</th>\n",
       "      <th>posteam_type</th>\n",
       "      <th>play_type</th>\n",
       "      <th>game_seconds_remaining</th>\n",
       "      <th>posteam_timeouts_remaining</th>\n",
       "      <th>PassingMean</th>\n",
       "      <th>RushingMean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>276133</th>\n",
       "      <td>4</td>\n",
       "      <td>45.0</td>\n",
       "      <td>9</td>\n",
       "      <td>390.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-21.0</td>\n",
       "      <td>DAL</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>390.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.095238</td>\n",
       "      <td>7.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276135</th>\n",
       "      <td>4</td>\n",
       "      <td>80.0</td>\n",
       "      <td>10</td>\n",
       "      <td>281.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-24.0</td>\n",
       "      <td>DAL</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>281.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.095238</td>\n",
       "      <td>7.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276136</th>\n",
       "      <td>4</td>\n",
       "      <td>71.0</td>\n",
       "      <td>1</td>\n",
       "      <td>255.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-24.0</td>\n",
       "      <td>DAL</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>255.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.095238</td>\n",
       "      <td>7.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276137</th>\n",
       "      <td>4</td>\n",
       "      <td>67.0</td>\n",
       "      <td>10</td>\n",
       "      <td>232.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-24.0</td>\n",
       "      <td>DAL</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>232.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.095238</td>\n",
       "      <td>7.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276138</th>\n",
       "      <td>4</td>\n",
       "      <td>57.0</td>\n",
       "      <td>10</td>\n",
       "      <td>203.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-24.0</td>\n",
       "      <td>DAL</td>\n",
       "      <td>home</td>\n",
       "      <td>pass</td>\n",
       "      <td>203.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.095238</td>\n",
       "      <td>7.666667</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        qtr  yardline_100  ydstogo  half_seconds_remaining  down  \\\n",
       "276133    4          45.0        9                   390.0   2.0   \n",
       "276135    4          80.0       10                   281.0   1.0   \n",
       "276136    4          71.0        1                   255.0   2.0   \n",
       "276137    4          67.0       10                   232.0   1.0   \n",
       "276138    4          57.0       10                   203.0   1.0   \n",
       "\n",
       "        score_differential posteam posteam_type play_type  \\\n",
       "276133               -21.0     DAL         home      pass   \n",
       "276135               -24.0     DAL         home      pass   \n",
       "276136               -24.0     DAL         home      pass   \n",
       "276137               -24.0     DAL         home      pass   \n",
       "276138               -24.0     DAL         home      pass   \n",
       "\n",
       "        game_seconds_remaining  posteam_timeouts_remaining  PassingMean  \\\n",
       "276133                   390.0                         3.0     5.095238   \n",
       "276135                   281.0                         1.0     5.095238   \n",
       "276136                   255.0                         1.0     5.095238   \n",
       "276137                   232.0                         1.0     5.095238   \n",
       "276138                   203.0                         1.0     5.095238   \n",
       "\n",
       "        RushingMean  \n",
       "276133     7.666667  \n",
       "276135     7.666667  \n",
       "276136     7.666667  \n",
       "276137     7.666667  \n",
       "276138     7.666667  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 8, 25,  9, 30, 23, 27, 28, 22,  1, 34, 24,  0, 20, 16,  3,  6, 10,\n",
       "        5,  7, 12, 14, 19, 33, 29,  4, 11,  2, 13, 21, 26, 32, 31, 17, 15,\n",
       "       18])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "le_posteam = LabelEncoder()\n",
    "df['posteam'] = le_posteam.fit_transform(df['posteam'])\n",
    "df[\"posteam\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "le_posteam_type = LabelEncoder()\n",
    "df['posteam_type'] = le_posteam_type.fit_transform(df['posteam_type'])\n",
    "df[\"posteam_type\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#le_play_type = LabelEncoder()\n",
    "#df['play_type'] = le_play_type.fit_transform(df['play_type'])\n",
    "#df[\"play_type\"].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testando qual tipo de machine learning seria o melhor para esse caso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape X_train (92633, 12)\n",
      "Shape X_test (23159, 12)\n",
      "Shape y_train (92633,)\n",
      "Shape y_test (23159,)\n"
     ]
    }
   ],
   "source": [
    "y = df['play_type']\n",
    "X = df.drop('play_type', axis=1)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)\n",
    "print(\"Shape X_train\", X_train.shape)\n",
    "print(\"Shape X_test\",X_test.shape)\n",
    "print(\"Shape y_train\",y_train.shape)\n",
    "print(\"Shape y_test\",y_test.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando GradientBoostingClassifier foi de 69.62% \n"
     ]
    }
   ],
   "source": [
    "gbc = GradientBoostingClassifier()\n",
    "gbc.fit(X_train, y_train)\n",
    "\n",
    "gbc_predictions = gbc.predict(X_test)\n",
    "\n",
    "print(f\"A precisao do teste usando GradientBoostingClassifier foi de {accuracy_score(y_test, gbc_predictions)*100:.2f}% \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando GaussianNB foi de 64.02% \n"
     ]
    }
   ],
   "source": [
    "gnb = GaussianNB()\n",
    "gnb.fit(X_train, y_train)\n",
    "\n",
    "gnb_predictions = gnb.predict(X_test)\n",
    "print(f\"A precisao do teste usando GaussianNB foi de {accuracy_score(y_test, gnb_predictions)*100:.2f}% \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando DecisionTreeClassifier foi de 62.94% \n"
     ]
    }
   ],
   "source": [
    "desc_tree = DecisionTreeClassifier()\n",
    "desc_tree.fit(X_train, y_train)\n",
    "\n",
    "dt_predictions = desc_tree.predict(X_test)\n",
    "print(f\"A precisao do teste usando DecisionTreeClassifier foi de {accuracy_score(y_test, dt_predictions)*100:.2f}% \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando RandomForestClassifier foi de 69.21% \n"
     ]
    }
   ],
   "source": [
    "random_forest = RandomForestClassifier(n_estimators=100)\n",
    "random_forest.fit(X_train, y_train)\n",
    "\n",
    "rf_predictions = random_forest.predict(X_test)\n",
    "print(f\"A precisao do teste usando RandomForestClassifier foi de {accuracy_score(y_test, rf_predictions)*100:.2f}% \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando LogisticRegression foi de 62.32% \n"
     ]
    }
   ],
   "source": [
    "log_reg = LogisticRegression()\n",
    "log_reg.fit(X_train, y_train)\n",
    "\n",
    "lr_predictions = log_reg.predict(X_test)\n",
    "print(f\"A precisao do teste usando LogisticRegression foi de {accuracy_score(y_test, lr_predictions)*100:.2f}% \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A precisao do teste usando DecisionTreeClassifier foi de 63.36% \n"
     ]
    }
   ],
   "source": [
    "desc_tree = DecisionTreeClassifier()\n",
    "desc_tree.fit(X_train, y_train)\n",
    "\n",
    "dt_predictions = desc_tree.predict(X_test)\n",
    "print(f\"A precisao do teste usando DecisionTreeClassifier foi de {accuracy_score(y_test, dt_predictions)*100:.2f}% \")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Teste para ver como seria o input de um usuario"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['4', '78', '8', '100', '2', '-7', 'DAL', 'home', '100', '1',\n",
       "        '5.3', '7.1']], dtype='<U32')"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = np.array([[4, 78, 8, 100, 2, -7, 'DAL', 'home', 100, 1, 5.3, 7.1]])\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  4. ,  78. ,   8. , 100. ,   2. ,  -7. ,   8. ,   1. , 100. ,\n",
       "          1. ,   5.3,   7.1]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X[:, 6] = le_posteam.transform(X[:,6])\n",
    "X[:, 7] = le_posteam_type.transform(X[:,7])\n",
    "X = X.astype(float)\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['pass'], dtype=object)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test = gbc.predict(X)\n",
    "y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plotando um grafico que mostra quais colunas sao mais importantes na hora de decidir uma jogada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='feature'>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAACVEAAAL0CAYAAAAPjOXHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/av/WaAAAACXBIWXMAAA9hAAAPYQGoP6dpAABzUklEQVR4nOzdf3DV1YH//1cCkoCUH5Y2ATdj8FcRFWilslSd6pg1uG5XOq1Vul2U6eKuHXZKs7VKV8FKPwMqtdipI6sdV92p1e2udXerQ39kTGdqUVes21+2ox1YrBBAV4jCSJTk+0e/xqYG9AbkKufxmHmPyfuee+55kxyC8vR9a3p7e3sDAAAAAAAAAABQqNpqLwAAAAAAAAAAAKCaRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRhlZ7AftDT09PNm7cmHe9612pqamp9nIAAAAAAAAAAIAq6+3tzQsvvJAJEyaktnbv95o6KCKqjRs3pqmpqdrLAAAAAAAAAAAA3maefvrp/Mmf/MlexxwUEdW73vWuJL+/4FGjRlV5NQAAAAAAAAAAQLV1dXWlqampry3am4Mionr1LfxGjRologIAAAAAAAAAAPq82hbtzd7f7A8AAAAAAAAAAOAgJ6ICAAAAAAAAAACKJqICAAAAAAAAAACKNrTaCwAAAAAAAAAAgH3R29ubV155Jbt37672UjjAhgwZkqFDh6ampmaf5hFRAQAAAAAAAADwjtXd3Z1NmzZl586d1V4KVTJixIiMHz8+w4YNG/QcIioAAAAAAAAAAN6Renp6sm7dugwZMiQTJkzIsGHD9vmORLxz9Pb2pru7O1u3bs26detyzDHHpLa2dlBziagAAAAAAAAAAHhH6u7uTk9PT5qamjJixIhqL4cqGD58eA455JD87//+b7q7u1NfXz+oeQaXXgEAAAAAAAAAwNvEYO8+xMFhf3z9fQcBAAAAAAAAAABFE1EBAAAAAAAAAMABdvrpp2fhwoXVXgb/v6HVXgAAAAAAAAAAAOxvzZffd8Bea/3ycyp+zj333JNDDjnkLVjNvuvo6MgZZ5yR559/PmPGjKn2cg4IERUAAAAAAAAAABxghx12WLWXMKCXX3652kuoCm/nBwAAAAAAAAAAB9gfvp1fc3NzvvzlL2fu3LkZOXJkjjjiiPznf/5ntm7dmnPPPTcjR47MlClT8uijj/Y9/7bbbsuYMWNy77335phjjkl9fX1aW1vz9NNP93udm266KUcddVSGDRuW973vffmXf/mXfo/X1NTkpptuyl/+5V/m0EMPzfz583PGGWckScaOHZuamppcdNFFSZLVq1fn1FNPzZgxY/Lud787f/EXf5Hf/va3fXOtX78+NTU1ueeee3LGGWdkxIgRmTp1atasWdPvNR988MGcfvrpGTFiRMaOHZvW1tY8//zzSZKenp4sW7YsEydOzPDhwzN16tT827/92375Nd8bERUAAAAAAAAAAFTZV7/61Zxyyin56U9/mnPOOSd//dd/nblz5+ZTn/pUHnvssRx11FGZO3duent7+56zc+fO/L//9/9yxx135MEHH8y2bdtywQUX9D3+ne98J5/97GfzD//wD/nFL36Rv/3bv828efPywAMP9Hvtq666Kh/96Efz85//PF/60pfy7//+70mS3/zmN9m0aVNuuOGGJMmOHTvS1taWRx99NO3t7amtrc1HP/rR9PT09JvvH//xH/P5z38+jz/+eI499tjMmTMnr7zySpLk8ccfz5lnnpnJkydnzZo1+fGPf5yPfOQj2b17d5Jk2bJlueOOO7Jq1ar88pe/zOc+97l86lOfyo9+9KP9/4v+B2p6//BX9h2qq6sro0ePzvbt2zNq1KhqLwcAAAAAAAAAgAPgpZdeyrp16zJx4sTU19f3e6z58vsO2DrWLz+n4uecfvrpmTZtWlauXJnm5uacdtppfXeJ6uzszPjx43PllVfm6quvTpI89NBDmTlzZjZt2pTGxsbcdtttmTdvXh566KHMmDEjSfLrX/86xx13XB5++OGcfPLJOeWUU3L88cfn5ptv7nvdT3ziE9mxY0fuu+/3vz41NTVZuHBhvvrVr/aN6ejoyBlnnJHnn38+Y8aM2eM1PPvss3nPe96Tn//85znhhBOyfv36TJw4Md/4xjfy6U9/Oknyq1/9Kscff3yeeOKJTJo0KZ/85CezYcOG/PjHP37dfLt27cphhx2WH/7wh5k5c2bf+b/5m7/Jzp07c+eddw64jj19H1TSFLkTFQAAAAAAAAAAVNmUKVP6Pm5oaEiSnHjiia87t2XLlr5zQ4cOzQc/+MG+zydNmpQxY8bkiSeeSJI88cQTOeWUU/q9zimnnNL3+KumT5/+ptb45JNPZs6cOTnyyCMzatSoNDc3J0k2bNiwx2sZP358v3W/eieqgTz11FPZuXNn/uzP/iwjR47sO+64445+bxv4Vhj6ls4OAAAAAAAAAAC8oUMOOaTv45qamj2e++O3ztsfDj300Dc17iMf+UiOOOKI3HLLLZkwYUJ6enpywgknpLu7u9+4va17+PDhe5z/xRdfTJLcd999Ofzww/s9VldX96bWOFjuRAUAAAAAAAAAAO9Ar7zySh599NG+z3/zm99k27ZtOe6445Ikxx13XB588MF+z3nwwQczefLkvc47bNiwJMnu3bv7zj333HP5zW9+kyuuuCJnnnlmjjvuuDz//PMVr3nKlClpb28f8LHJkyenrq4uGzZsyNFHH93vaGpqqvi1KuFOVAAAAAAAAAAA8A50yCGH5O///u/zta99LUOHDs2CBQvyp3/6pzn55JOTJJdeemk+8YlP5P3vf39aWlryX//1X7nnnnvywx/+cK/zHnHEEampqcl3v/vd/Pmf/3mGDx+esWPH5t3vfnduvvnmjB8/Phs2bMjll19e8ZoXLVqUE088MZ/5zGfyd3/3dxk2bFgeeOCBnHfeeRk3blw+//nP53Of+1x6enpy6qmnZvv27XnwwQczatSoXHjhhYP6dXoz3IkKAAAAAAAAAADegUaMGJHLLrssn/zkJ3PKKadk5MiRufvuu/senz17dm644YasWLEixx9/fP7pn/4p//zP/5zTTz99r/Mefvjh+dKXvpTLL788DQ0NWbBgQWpra3PXXXdl7dq1OeGEE/K5z30u1113XcVrPvbYY/P9738///M//5OTTz45M2fOzH/8x39k6NDf3wtq6dKlufLKK7Ns2bIcd9xxmTVrVu67775MnDix4teqRE1vb2/vW/oKB0BXV1dGjx6d7du3Z9SoUdVeDgAAAAAAAAAAB8BLL72UdevWZeLEiamvr6/2cg6o2267LQsXLsy2bduqvZSq29P3QSVNkTtRAQAAAAAAAAAARRNRAQAAAAAAAAAARRNRAQAAAAAAAADAO8xFF13krfz2IxEVAAAAAAAAAABQNBEVAAAAAAAAAABQNBEVAAAAAAAAAADvaL29vdVeAlW0P77+IioAAAAAAAAAAN6RDjnkkCTJzp07q7wSqunVr/+r3w+DMXR/LQYAAAAAAAAAAA6kIUOGZMyYMdmyZUuSZMSIEampqanyqjhQent7s3PnzmzZsiVjxozJkCFDBj2XiAoAAAAAAAAAgHesxsbGJOkLqSjPmDFj+r4PBktE9TbSfPl91V5C8dYvP6faSwAAAAAAAAAAKlBTU5Px48fnve99b15++eVqL4cD7JBDDtmnO1C9SkQFAAAAAAAAAMA73pAhQ/ZLTEOZaqu9AAAAAAAAAAAAgGoSUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUTUQEAAAAAAAAAAEUbWu0FAPyh5svvq/YSird++TnVXgIAAAAAAAAAHFDuRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRNRAUAAAAAAAAAABRtUBHVjTfemObm5tTX12fGjBl55JFH9jj2lltuyWmnnZaxY8dm7NixaWlped34iy66KDU1Nf2OWbNmDWZpAAAAAAAAAAAAFak4orr77rvT1taWJUuW5LHHHsvUqVPT2tqaLVu2DDi+o6Mjc+bMyQMPPJA1a9akqakpZ511Vp555pl+42bNmpVNmzb1Hd/61rcGd0UAAAAAAAAAAAAVqDiiuv766zN//vzMmzcvkydPzqpVqzJixIjceuutA47/5je/mc985jOZNm1aJk2alG984xvp6elJe3t7v3F1dXVpbGzsO8aOHTu4KwIAAAAAAAAAAKhARRFVd3d31q5dm5aWltcmqK1NS0tL1qxZ86bm2LlzZ15++eUcdthh/c53dHTkve99b973vvflkksuyXPPPbfHOXbt2pWurq5+BwAAAAAAAAAAwGBUFFE9++yz2b17dxoaGvqdb2hoSGdn55ua47LLLsuECRP6hVizZs3KHXfckfb29lxzzTX50Y9+lLPPPju7d+8ecI5ly5Zl9OjRfUdTU1MllwEAAAAAAAAAANBn6IF8seXLl+euu+5KR0dH6uvr+85fcMEFfR+feOKJmTJlSo466qh0dHTkzDPPfN08ixYtSltbW9/nXV1dQioAAAAAAAAAAGBQKroT1bhx4zJkyJBs3ry53/nNmzensbFxr89dsWJFli9fnu9///uZMmXKXsceeeSRGTduXJ566qkBH6+rq8uoUaP6HQAAAAAAAAAAAINRUUQ1bNiwnHTSSWlvb+8719PTk/b29sycOXOPz7v22muzdOnSrF69OtOnT3/D1/nd736X5557LuPHj69keQAAAAAAAAAAABWrKKJKkra2ttxyyy25/fbb88QTT+SSSy7Jjh07Mm/evCTJ3Llzs2jRor7x11xzTa688srceuutaW5uTmdnZzo7O/Piiy8mSV588cVceumleeihh7J+/fq0t7fn3HPPzdFHH53W1tb9dJkAAAAAAAAAAAADG1rpE84///xs3bo1ixcvTmdnZ6ZNm5bVq1enoaEhSbJhw4bU1r7WZt10003p7u7Oxz/+8X7zLFmy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pkouAwAAAAAAAAAAoE/Fb+e3L5YvX5677ror3/nOd1JfXz/oeRYtWpTt27f3HU8//fR+XCUAAAAAAAAAAFCSoZUMHjduXIYMGZLNmzf3O7958+Y0Njbu9bkrVqzI8uXL88Mf/jBTpkzpO//q8zZv3pzx48f3m3PatGkDzlVXV5e6urpKlg4AAAAAAAAAADCgiu5ENWzYsJx00klpb2/vO9fT05P29vbMnDlzj8+79tprs3Tp0qxevTrTp0/v99jEiRPT2NjYb86urq48/PDDe50TAAAAAAAAAABgf6joTlRJ0tbWlgsvvDDTp0/PySefnJUrV2bHjh2ZN29ekmTu3Lk5/PDDs2zZsiTJNddck8WLF+fOO+9Mc3NzOjs7kyQjR47MyJEjU1NTk4ULF+bLX/5yjjnmmEycODFXXnllJkyYkNmzZ++/KwUAAAAAAAAAABhAxRHV+eefn61bt2bx4sXp7OzMtGnTsnr16jQ0NCRJNmzYkNra125wddNNN6W7uzsf//jH+82zZMmSXHXVVUmSL3zhC9mxY0cuvvjibNu2LaeeempWr16d+vr6fbg0AAAAAAAAAACAN1bT29vbW+1F7Kuurq6MHj0627dvz6hRo6q9nEFrvvy+ai+heOuXn1PtJRTPPqg++wAAAAAAAACAg0ElTVHtXh8FAAAAAAAAAAA4yImoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAoomoAAAAAAAAAACAog0qorrxxhvT3Nyc+vr6zJgxI4888sge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VSqOqJJkwYIFWbBgwYCPvRpGvaq5uTm9vb37NCcAAAAAAAAAAMBbpfaNhwAAAAAAAAAAABy8RFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAAAAAAAEDRRFQAAAAA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r17FhQsXKh1naGjY4L6nTPWvIedHenBdvHgRV65cqXSchYUF7Ozs6jhFdefWrVs4d+5clePd3NxgYMAXKdOji3GJiIiIiEh3f/31F/76669Kx5mamqJJkyZ1nKL6dePGDfz+++9Vjvfw8KjD1BAR0b1gjKt/7ERFREREREREREREREREREREREQNGj/nR0REREREREREREREREREREREDRo7URERERERERERERERERERERERUYPGTlRERERERERERERERERERERERNSgsRMVERERERERERHVORHByy+/DGtra6hUKqSnp9d3koiIiIiIiIiIqAFTiYjUdyKIiIiIiIiIiKhh2bt3L8LDw5GcnIxmzZqhcePGMDAwuKdlRkRE4PLly/jss89qJ5FERERERERERNRg3FvLFBERERERERER0V3IycmBo6MjOnfuXN9JqaC4uBgqlQp6enyJOxERERERERFRQ8GWICIiIiIiIiIiqlMREREYP3488vLyoFKp4ObmhpKSEixYsADu7u4wNTWFv78/Pv30U2We4uJiREZGKuO9vb2xbNkyZfysWbPw0Ucf4fPPP4dKpYJKpUJycjKSk5OhUqlw+fJlZdr09HSoVCqcO3cOABAfHw8rKyvs2rULLVu2hLGxMfLy8lBYWIjo6Gg0adIE5ubm6NixI5KTk+toLxERERERERERUV3im6iIiIiIiIiIiKhOLVu2DM2bN8fq1auRlpYGfX19LFiwAB9//DFWrlwJT09PHDx4EM8//zxsbW0RHByMkpISNG3aFNu2bYONjQ1SU1Px8ssvw9HREYMGDUJ0dDQyMjJw5coVxMXFAQCsra2RmpqqU5quX7+OhQsXYu3atbCxsYGdnR3GjRuHU6dOYcuWLXBycsLOnTvRp08fnDhxAp6envdzFxERERERERERUR1jJyoiIiIiIiIiIqpTlpaW0Gg00NfXh4ODAwoLC/H2228jKSkJgYGBAIBmzZrh0KFDWLVqFYKDg2FoaIjZs2cry3B3d8fhw4fxySefYNCgQVCr1TA1NUVhYSEcHBzuOE1FRUV4//334e/vDwDIy8tDXFwc8vLy4OTkBACIjo7Gvn37EBcXh7fffrsW9gQRERERERERET0o2ImKiIiIiIiIiIjqVXZ2Nq5fv45evXppDb958yYCAgKU3ytWrMC6deuQl5eHGzdu4ObNm2jdunWtpMHIyAitWrVSfp84cQLFxcXw8vLSmq6wsBA2Nja1sk4iIiIiIiIiInpwsBMVERERERERERHVq4KCAgDA7t270aRJE61xxsbGAIAtW7YgOjoasbGxCAwMhEajQUxMDI4cOVLtsvX09AAAIqIMKyoqqjCdqakpVCqVVpr09fXx448/Ql9fX2tatVp9B1tHREREREREREQPA3aiIiIiIiIiIiKietWyZUsYGxsjLy8PwcHBlU6TkpKCzp07Y8yYMcqwnJwcrWmMjIxQXFysNczW1hYAcP78eTRq1AgAkJ6eXmOaAgICUFxcjIsXL6Jr1653sjlERERERERERPQQYicqIiIiIiIiIiKqVxqNBtHR0Xj11VdRUlKCLl264O+//0ZKSgosLCwwfPhweHp6Yv369di/fz/c3d2xYcMGpKWlwd3dXVmOm5sb9u/fj6ysLNjY2MDS0hIeHh5wdnbGrFmzMH/+fJw+fRqxsbE1psnLywtDhw7FsGHDEBsbi4CAAPz555/46quv0KpVK/Tr1+9+7hIiIiIiIiIiIqpjevWdACIiIiIiIiIiorlz52L69OlYsGABfHx80KdPH+zevVvpJPXKK69g4MCBGDx4MDp27Ij8/Hytt1IBwMiRI+Ht7Y127drB1tYWKSkpMDQ0xObNm5GZmYlWrVph4cKFmDdvnk5piouLw7BhwzBp0iR4e3tjwIABSEtLg4uLS61vPxERERERERER1S+ViEh9J4KIiIiIiIiIiIiIiIiIiIiIiKi+8E1URERERERERERERERERERERETUoLETFRERERERERERERERERERERERNWjsREVERERERERERERERERERERERA0aO1EREREREREREREREREREREREVGDxk5URERERERERERERERERERERETUoLETFRERERERERERERERERERERERNWjsREVERERERERERERERERERERERA0aO1EREREREREREREREREREREREVGDxk5URERERERERERERERERERERETUoLETFRERERERERERERERERERERERNWjsREVERERERERERERERERERERERA0aO1EREREREREREREREREREREREVGD9v8Aa9aT+VQMlcoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 3000x900 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(gbc.feature_importances_,3)})\n",
    "importances = importances.sort_values('importance',ascending=False).set_index('feature')\n",
    "importances.plot.bar(figsize=(30,9),rot=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Salvando as funcoes criadas com o pickle para integrar com o site"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {\"model\": gbc, \"le_posteam\": le_posteam, \"le_posteam_type\": le_posteam_type}\n",
    "with open('saved_steps.pkl', 'wb') as file:\n",
    "    pickle.dump(data, file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('saved_steps.pkl', 'rb') as file:\n",
    "    data = pickle.load(file)\n",
    "\n",
    "gbc = data[\"model\"]\n",
    "le_posteam = data[\"le_posteam\"]\n",
    "le_posteam_type = data[\"le_posteam_type\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['pass'], dtype=object)"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test = gbc.predict(X)\n",
    "y_test"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10.8 64-bit (microsoft store)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "e9e375da22a79186ffe49072ae6df2c8be50fc2c627f3d12cad18e70942528a3"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
