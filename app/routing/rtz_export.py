"""
Maritime Route Exporters: IEC 61174 RTZ and GPX.

Exports planned Antarctic routes into standard maritime formats:
1. IEC 61174 (Edition 4) Route Plan Exchange format (.rtz XML) compatible with ECDIS.
2. GPS Exchange Format (.gpx XML) compatible with GPS handhelds and GIS software.
"""

from typing import List, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime


class MaritimeRouteExporter:
    """
    Standard route serialization for maritime navigation systems.
    """

    @staticmethod
    def export_rtz(
        route_name: str,
        waypoints: List[Tuple[float, float]],
        vessel_name: str = "RV Bharati Explorer",
        speed_kts: float = 12.0
    ) -> str:
        """
        Generate IEC 61174 Edition 4 RTZ XML.

        Args:
            route_name: Name of route.
            waypoints: List of (lat, lon) tuples.
            vessel_name: Vessel name identifier.
            speed_kts: Default planned transit speed.

        Returns:
            Prettified XML string compliant with RTZ schema.
        """
        # Root element
        root = ET.Element(
            "route",
            attrib={
                "version": "1.0",
                "xmlns": "http://www.cirm.org/RTZ/1/0",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://www.cirm.org/RTZ/1/0 rtz.xsd"
            }
        )

        # Route info
        route_info = ET.SubElement(
            root,
            "routeInfo",
            attrib={
                "routeName": route_name,
                "vesselName": vessel_name,
                "author": "Polar Navigator AI (SIH Prototype)",
                "generationTime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        )

        # Waypoints container
        wp_container = ET.SubElement(root, "waypoints")

        for idx, (lat, lon) in enumerate(waypoints, start=1):
            wp = ET.SubElement(
                wp_container,
                "waypoint",
                attrib={
                    "id": str(idx),
                    "name": f"WP_{idx:03d}",
                    "radius": "0.5"
                }
            )
            # Position: IEC 61174 uses WGS84 lat and lon
            ET.SubElement(
                wp,
                "position",
                attrib={
                    "lat": f"{lat:.6f}",
                    "lon": f"{lon:.6f}"
                }
            )
            # Leg parameters
            if idx < len(waypoints):
                ET.SubElement(
                    wp,
                    "leg",
                    attrib={
                        "portsideXTD": "0.5",
                        "starboardXTD": "0.5",
                        "safetyContour": "30.0",
                        "geometryType": "Orthodromic",  # Great circle geodesic
                        "speedMin": f"{max(4.0, speed_kts - 4.0):.1f}",
                        "speedMax": f"{speed_kts + 2.0:.1f}"
                    }
                )

        rough_string = ET.tostring(root, encoding="utf-8")
        parsed = minidom.parseString(rough_string)
        return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    @staticmethod
    def export_gpx(
        route_name: str,
        waypoints: List[Tuple[float, float]],
        description: str = "Polar Navigator AI Optimized Route"
    ) -> str:
        """
        Generate standard GPX XML format for GIS and GPS devices.
        """
        root = ET.Element(
            "gpx",
            attrib={
                "version": "1.1",
                "creator": "Polar Navigator AI - SIH MoES/NCPOR Prototype",
                "xmlns": "http://www.topografix.com/GPX/1/1"
            }
        )

        metadata = ET.SubElement(root, "metadata")
        name_elem = ET.SubElement(metadata, "name")
        name_elem.text = route_name
        desc_elem = ET.SubElement(metadata, "desc")
        desc_elem.text = description
        time_elem = ET.SubElement(metadata, "time")
        time_elem.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        rte = ET.SubElement(root, "rte")
        rte_name = ET.SubElement(rte, "name")
        rte_name.text = route_name

        for idx, (lat, lon) in enumerate(waypoints, start=1):
            rtept = ET.SubElement(
                rte,
                "rtept",
                attrib={
                    "lat": f"{lat:.6f}",
                    "lon": f"{lon:.6f}"
                }
            )
            pt_name = ET.SubElement(rtept, "name")
            pt_name.text = f"WP{idx:03d}"

        rough_string = ET.tostring(root, encoding="utf-8")
        parsed = minidom.parseString(rough_string)
        return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
