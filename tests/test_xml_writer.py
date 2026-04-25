import unittest
from xml.etree import ElementTree as ET

from uiir.models import BBox, UIIRDocument, UINode
from uiir.xml_writer import document_to_xml_element


class XMLWriterTests(unittest.TestCase):
    def test_node_attrs_include_required_fields(self):
        doc = UIIRDocument(
            version="0.1",
            source="sample.psd",
            width=1920,
            height=1080,
            assets_root="assets/",
            root=UINode(
                id="n1",
                type="Screen",
                bbox=BBox(0, 0, 1920, 1080),
                confidence=1,
                source_refs=["document"],
                children=[
                    UINode(
                        id="n2",
                        type="Button",
                        bbox=BBox(10, 20, 100, 40),
                        confidence=0.9,
                        source_refs=["layer:1"],
                        text="OK",
                    )
                ],
            ),
        )
        xml = ET.tostring(document_to_xml_element(doc), encoding="unicode")
        self.assertIn('type="Button"', xml)
        self.assertIn('bbox="10,20,100,40"', xml)
        self.assertIn('sourceRefs="layer:1"', xml)
        self.assertIn('text="OK"', xml)


if __name__ == "__main__":
    unittest.main()
