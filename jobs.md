---
title: "Collaborate with us"
author: "Felice Pantaleo"
layout: default
markdown: kramdown

---

# Work with us

<ul>
  {% for page in site.pages %}
    {% if page.resource == true %}
      {% for pc in page.categories %}
        {% if pc == "jobs-ads" %}
          <li><a href="{{site.baseurl}}/{{ page.url }}">{{ page.title }}: {{ page.description }}</a></li>
        {% endif %}   <!-- cat-match-p -->
      {% endfor %}  <!-- page-category -->
    {% endif %}   <!-- resource-p -->
  {% endfor %}  <!-- page -->
</ul>
