---
title: "Events"
author: "Felice Pantaleo"
layout: default
---

# Attended events

{% for page in site.pages reversed%}{% if page.resource == true %}{% for pc in page.categories %}{% if pc == "events" %}- [{{ page.date }},  {{ page.title }}]({{site.baseurl}}/{{page.url }})

{% endif %}{% endfor %}{% endif %}{% endfor %}  
