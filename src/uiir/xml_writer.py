from __future__ import annotations

import json
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from .models import UIIRDocument, UINode


def document_to_xml_element(document: UIIRDocument) -> ET.Element:
    root = ET.Element(
        "UIIR",
        {
            "version": document.version,
            "source": document.source,
            "width": str(document.width),
            "height": str(document.height),
        },
    )
    ET.SubElement(root, "Assets", {"root": document.assets_root})
    root.append(_node_to_element(document.root))
    return root


def write_xml(document: UIIRDocument, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = ET.tostring(document_to_xml_element(document), encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")
    output.write_bytes(pretty)
    return output


def write_json(document: UIIRDocument, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _node_to_element(node: UINode) -> ET.Element:
    attrs = {
        "id": node.id,
        "type": node.type,
        "bbox": node.bbox.to_attr(),
        "confidence": f"{float(node.confidence):.3f}",
        "sourceRefs": ",".join(node.source_refs),
    }
    optional = {
        "role": node.role,
        "text": node.text,
        "style": node.style,
        "layout": node.layout,
        "asset": node.asset,
        "interaction": node.interaction,
    }
    attrs.update({key: value for key, value in optional.items() if value})
    metadata_attrs = {
        "psdParentId": node.metadata.get("psdParentId"),
        "psdPath": node.metadata.get("psdPath"),
        "psdDepth": node.metadata.get("psdDepth"),
        "groupingReason": node.metadata.get("groupingReason"),
    }
    attrs.update({key: str(value) for key, value in metadata_attrs.items() if value is not None and value != ""})
    element = ET.Element("Node", attrs)
    for child in node.children:
        element.append(_node_to_element(child))
    return element
