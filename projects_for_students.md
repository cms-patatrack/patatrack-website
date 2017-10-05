---
title: "Projects for students"
author: "Felice Pantaleo"
layout: default
markdown: kramdown

---

# Projects for students
Patatrack is currently offering Bachelor, Master and PhD level theses projects in Particle Physics, Computer Science, Computational Physics, Computing Engineering.

<ul>
  {% for page in site.pages %}
    {% if page.resource == true %}
      {% for pc in page.categories %}
        {% if pc == "students-projects" %}
          <li><a href="{{site.baseurl}}/{{ page.url }}">{{ page.title }}</a></li>
        {% endif %}   <!-- cat-match-p -->
      {% endfor %}  <!-- page-category -->
    {% endif %}   <!-- resource-p -->
  {% endfor %}  <!-- page -->
</ul>
