{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   .. rubric:: 模块内容

   .. autosummary::
      :toctree:
      :template: autosummary/class.rst

      {% for item in classes %}
      {{ item }}
      {%- endfor %}

   .. autosummary::
      :toctree:

      {% for item in functions %}
      {{ item }}
      {%- endfor %}