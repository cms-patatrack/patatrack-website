---
title: "Events"
author: "Felice Pantaleo"
layout: default
---

# Events

Patatrack is a collaboration between the following institutes and people:

{% for page in site.pages %}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "organization" %}[{{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}  
