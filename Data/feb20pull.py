# %%
import pandas as pd
from API import get_book, baseurl
dataset = pd.read_csv('american19thcenturyLit.csv')

downloads = []
for i in range(len(dataset)):
    downloaded = get_book(dataset.loc[i,'book'], dataset.loc[i,'author'], baseurl)
    downloads.append(downloaded)

print(downloads)

# %%

downloads_df = pd.DataFrame(
[{'id': 32325,
  'title': "The Adventures of Huckleberry Finn (Tom Sawyer's Comrade)"},
 {'id': 25344, 'title': 'The Scarlet Letter'},
 {'id': 37106, 'title': 'Little Women; Or, Meg, Jo, Beth, and Amy'},
 {'id': 41, 'title': 'The Legend of Sleepy Hollow'},
 {'id': 73,
  'title': 'The Red Badge of Courage: An Episode of the American Civil War'},
 {'id': 160, 'title': 'The Awakening, and Selected Short Stories'},
 {'id': 205, 'title': 'Walden, and On The Duty Of Civil Disobedience'},
 {'id': 74, 'title': 'The Adventures of Tom Sawyer, Complete'},
 {'id': 203, 'title': "Uncle Tom's Cabin"},
 {'id': 209, 'title': 'The Turn of the Screw'},
 {'id': 284, 'title': 'The House of Mirth'},
 {'id': 17192, 'title': 'The Raven'},
 {'id': 1322, 'title': 'Leaves of Grass'},
 {'id': 165, 'title': 'McTeague: A Story of San Francisco'},
 {'id': 367, 'title': 'The Country of the Pointed Firs'},
 {'id': 77, 'title': 'The House of the Seven Gables'},
 {'id': 102, 'title': "The Tragedy of Pudd'nhead Wilson"},
 {'id': 447, 'title': 'Maggie: A Girl of the Streets'},
 {'id': 2870, 'title': 'Washington Square'},
 {'id': 208, 'title': 'Daisy Miller: A Study'},
 {'id': 86, 'title': "A Connecticut Yankee in King Arthur's Court"},
 {'id': 541, 'title': 'The Age of Innocence'},
 {'id': 205, 'title': 'Walden, and On The Duty Of Civil Disobedience'},
 {'id': 2679, 'title': 'Poems by Emily Dickinson, Series Two'},
 {'id': 12352, 'title': 'Iola Leroy; Or, Shadows Uplifted'},
 {'id': 154, 'title': 'The Rise of Silas Lapham'},
 {'id': 8492, 'title': 'The King in Yellow'},
 {'id': 3176, 'title': 'The Innocents Abroad'},
 {'id': 65636, 'title': 'A Lost Lady'},
 {'id': 4600, 'title': 'A Hazard of New Fortunes — Complete'},
 {'id': 140, 'title': 'The Jungle'},
 {'id': 11228, 'title': 'The Marrow of Tradition'},
 {'id': 25439, 'title': 'Looking Backward: 2000-1887'},
 {'id': 35538,
  'title': 'The Squatter and the Don: A Novel Descriptive of Contemporary Occurrences in California'},
 {'id': 584,
  'title': "Our Nig; Or, Sketches from the Life of a Free Black, in a Two-story White House, North: Showing That Slavery's Shadows Fall Even There"},
 {'id': 22444, 'title': 'Selected Poems of Emily Dickinson'},
 {'id': 12718, 'title': 'Song of Myself Selections'},
 {'id': 233, 'title': 'Sister Carrie: A Novel'},
 {'id': 432, 'title': 'The Ambassadors'},
 {'id': 29452, 'title': 'The Wings of the Dove, Volume 1 of 2'},
 {'id': 2788, 'title': "Little Men: Life at Plumfield With Jo's Boys"},
 {'id': 2788, 'title': "Little Men: Life at Plumfield With Jo's Boys"},
 {'id': 2833, 'title': 'The Portrait of a Lady — Volume 1'},
 {'id': 45631,
  'title': 'Twelve Years a Slave: Narrative of Solomon Northup, a Citizen of New-York, Kidnapped in Washington City in 1841, and Rescued in 1853, from a Cotton Plantation near the Red River in Louisiana'},
 {'id': 45524, 'title': 'The Open Boat and Other Stories'},
 {'id': 3177, 'title': 'Roughing It'},
 {'id': 245, 'title': 'Life on the Mississippi'},
 {'id': 1837, 'title': 'The Prince and the Pauper'},
 {'id': 3186, 'title': 'The Mysterious Stranger, and Other Stories'},
 {'id': 23,
  'title': 'Narrative of the Life of Frederick Douglass, an American Slave'},
 {'id': 11030,
  'title': 'Incidents in the Life of a Slave Girl, Written by Herself'},
 {'id': 60976, 'title': 'Rip Van Winkle'},
 {'id': 2081, 'title': 'The Blithedale Romance'}])
# %%
downloads_df
# %%
