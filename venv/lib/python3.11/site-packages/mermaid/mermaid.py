import uuid

class Mermaid:
    def __init__(self, diagram: str):
        self._diagram = self._process_diagram(diagram)
        self._uid = uuid.uuid4()

    @staticmethod
    def _process_diagram(diagram: str) -> str:
        _diagram = diagram.replace("\n", "\\n")
        _diagram = _diagram.lstrip("\\n")
        _diagram = _diagram.replace("'", '"')
        return _diagram

    def _repr_html_(self) -> str:
        ret = f"""
        <div class="mermaid-{self._uid}"></div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.1.0/+esm'
            const graphDefinition = \'_diagram\';
            const element = document.querySelector('.mermaid-{self._uid}');
            const {{ svg }} = await mermaid.render('graphDiv-{self._uid}', graphDefinition);
            element.innerHTML = svg;
        </script>
        """
        ret = ret.replace("_diagram", self._diagram)
        return ret
