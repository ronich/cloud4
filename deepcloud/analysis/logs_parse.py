
# coding: utf-8

# In[1]:


import re
import os
import json


# In[2]:


files = []
for file in os.listdir('../logs'):
    if re.match('^20170610.*log', file):
        files.append(file)


# In[7]:


def parseLog(file):
    with open('../analysis/data/{}.jsons'.format(file[:-4]), 'w') as fw:
        with open('../logs/{}'.format(file)) as f:
            for line in f:
                if re.match('^\{', line):
                    json.dump(json.loads(line.replace("'", '"')), fw)
                    fw.write('\n')


# In[8]:


[parseLog(file) for file in files]


# In[29]:


print(file)


# In[ ]:




