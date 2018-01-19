---
title: "Wiki"
author: "Felice Pantaleo"
layout: wiki
---

## {{page.title}}


### Accelerated Pixel Tracks
{% for page in site.pages %}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "wiki" %}{% for act in page.activity %}{% if act == "pixeltracks" %}[{{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}{% endif %}{% endfor %}  

### Heterogeneous Computing
{% for page in site.pages %}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "wiki" %}{% for act in page.activity %}{% if act == "heterogeneouscomputing" %}[{{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}{% endif %}{% endfor %}  

### Machine Learning
{% for page in site.pages %}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "wiki" %}{% for act in page.activity %}{% if act == "ml" %}[{{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}{% endif %}{% endfor %}  



### 2nd Patatrack Hackathon
{% for page in site.pages %}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "wiki" %}{% for act in page.activity %}{% if act == "hackathon" %}[{{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}{% endif %}{% endfor %}  
