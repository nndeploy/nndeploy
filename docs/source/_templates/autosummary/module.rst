{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   
   {% if classes %}
   .. rubric:: Classes
   
   .. autosummary::
      :toctree:
      
      {% for class in classes %}
      {{ class }}
      {% endfor %}
   {% endif %}
   
   {% if functions %}
   .. rubric:: Functions
   
   .. autosummary::
      :toctree:
      
      {% for function in functions %}
      {{ function }}
      {% endfor %}
   {% endif %}