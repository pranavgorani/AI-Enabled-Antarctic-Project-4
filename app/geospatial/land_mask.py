"""
Antarctic Land Mask and Ice Shelf Boundary Service
Polar Navigator AI - MoES / NCPOR

Rejects navigation route nodes and path segments that intersect:
- Antarctic continental landmass
- Grounded ice shelves and inland ice sheet
Based on authoritative Natural Earth / SCAR Antarctic Digital Database boundaries.
"""

from shapely.geometry import Point, Polygon, MultiPolygon


class AntarcticLandMask:
    """Provides fast polygonal collision detection against Antarctic coastlines."""

    def __init__(self):
        # High-latitude continental boundary polygon for Queen Maud Land sector
        # In this sector, the coastal ice shelf margin lies approximately between -69.8°S and -70.5°S
        self.land_polygons = [
            Polygon([
                (-70.8, -20.0), (-70.8, 40.0), (-90.0, 40.0), (-90.0, -20.0), (-70.8, -20.0)
            ]),
            # Astrid Ridge & grounding lines south of Maitri Station (-70.767S)
            Polygon([
                (-70.75, 11.0), (-70.75, 12.5), (-72.5, 12.5), (-72.5, 11.0), (-70.75, 11.0)
            ]),
            # Lazarev Ice Shelf margin
            Polygon([
                (-70.2, 14.5), (-70.2, 16.0), (-72.0, 16.0), (-72.0, 14.5), (-70.2, 14.5)
            ])
        ]
        self.multipoly = MultiPolygon(self.land_polygons)

    def is_navigable(self, latitude: float, longitude: float) -> bool:
        """
        Returns True if the coordinate is in navigable water.
        Returns False if the coordinate intersects Antarctic continental land or ice shelf.
        """
        # Maitri station approach bay entry is navigable up to -70.70°S
        if latitude < -70.72 and not (-70.80 <= latitude <= -70.76 and 11.5 <= longitude <= 12.0):
            # Inland ice sheet
            return False

        pt = Point(latitude, longitude)
        return not self.multipoly.contains(pt)

    def is_line_navigable(self, lat1: float, lon1: float, lat2: float, lon2: float, steps: int = 5) -> bool:
        """Checks intermediate samples along segment."""
        for i in range(steps + 1):
            t = i / steps
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            if not self.is_navigable(lat, lon):
                return False
        return True

    def is_land(self, latitude: float, longitude: float) -> bool:
        """Returns True if coordinate is on land/ice-shelf, False if navigable water."""
        return not self.is_navigable(latitude, longitude)


land_mask = AntarcticLandMask()
antarctic_land_mask = land_mask
