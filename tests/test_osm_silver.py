"""pyosmium parser on a tiny hand-written OSM file: POIs classified by the taxonomy,
roads + junction street counts, buildings, land use, node/outline duplicates removed."""

import geopandas as gpd
from shapely.geometry import box

from geomlab.features import osm_silver
from geomlab.ingestion import osm as osm_ingest

OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6">
  <node id="1" lat="33.5800" lon="-7.5900"/>
  <node id="2" lat="33.5800" lon="-7.5800"/>
  <node id="3" lat="33.5900" lon="-7.5800"/>
  <node id="4" lat="33.5700" lon="-7.5800"/>
  <node id="10" lat="33.5850" lon="-7.5850"><tag k="amenity" v="pharmacy"/><tag k="name" v="Pharmacie A صيدلية"/><tag k="name:fr" v="Pharmacie A"/></node>
  <node id="11" lat="33.5851" lon="-7.5851"><tag k="amenity" v="school"/></node>
  <node id="12" lat="33.5852" lon="-7.5852"><tag k="highway" v="bus_stop"/></node>
  <node id="13" lat="40.0" lon="2.0"><tag k="amenity" v="cafe"/></node>
  <node id="20" lat="33.5830" lon="-7.5870"/>
  <node id="21" lat="33.5830" lon="-7.5868"/>
  <node id="22" lat="33.5832" lon="-7.5868"/>
  <node id="23" lat="33.5832" lon="-7.5870"/>
  <node id="24" lat="33.58305" lon="-7.58695"><tag k="shop" v="bakery"/></node>
  <node id="30" lat="33.5860" lon="-7.5860"><tag k="place" v="neighbourhood"/><tag k="name" v="Guéliz ⴳⵉⵍⵉⵣ گليز"/></node>
  <node id="31" lat="33.5870" lon="-7.5870"><tag k="place" v="suburb"/><tag k="name" v="دوار الحاجة"/><tag k="name:en" v="Douar Hajja"/></node>
  <node id="32" lat="33.5880" lon="-7.5880"><tag k="place" v="city"/><tag k="name" v="Big City"/></node>
  <node id="33" lat="33.5890" lon="-7.5890"><tag k="place" v="quarter"/></node>
  <node id="34" lat="35.0" lon="-5.0"><tag k="place" v="quarter"/><tag k="name" v="Elsewhere"/></node>
  <way id="100"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/></way>
  <way id="101"><nd ref="3"/><nd ref="2"/><nd ref="4"/><tag k="highway" v="residential"/></way>
  <way id="102"><nd ref="20"/><nd ref="21"/><nd ref="22"/><nd ref="23"/><nd ref="20"/>
    <tag k="building" v="yes"/><tag k="shop" v="bakery"/></way>
  <way id="103"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="landuse" v="residential"/></way>
</osm>
"""


def test_osm_silver_parses_and_classifies(env, monkeypatch):
    from geomlab.lakehouse import get_lake

    path = env / "tiny.osm"
    path.write_text(OSM)
    lake = get_lake()
    lake.write("silver", "city_boundary",
               gpd.GeoDataFrame({"city": ["testville"]}, geometry=[box(-7.60, 33.56, -7.57, 33.60)], crs=4326))
    monkeypatch.setattr(osm_ingest, "ensure_working_copy", lambda lake=None: path)

    counts = osm_silver.run(["testville"], lake)
    poi = lake.read("silver", "poi")
    cats = dict(zip(poi["category"], poi["role"], strict=True))
    assert cats == {"pharmacy": "business", "school": "attractor", "bus_stop": "attractor", "bakery": "business"}
    assert counts["testville"] == 4  # bakery node + bakery outline deduplicated, Paris café outside the bbox
    nodes = lake.read("silver", "road_nodes").set_index("node_id")
    assert nodes.loc[2, "street_count"] == 3  # end of the primary + middle of the residential way → junction
    assert set(lake.read("silver", "roads")["road_class"]) == {"primary", "local"}
    assert len(lake.read("silver", "buildings")) == 1
    assert lake.read("silver", "buildings")["footprint_m2"].iloc[0] > 300
    assert list(lake.read("silver", "landuse")["landuse_class"]) == ["residential"]
    places = lake.read("silver", "places").set_index("osm_id")
    assert places["name"].to_dict() == {"n30": "Guéliz", "n31": "Douar Hajja"}  # city, unnamed, outside: dropped
    assert places.loc["n31", "place"] == "suburb"
    ph = poi[poi["category"] == "pharmacy"].iloc[0]
    assert (ph["name"], ph["name_fr"]) == ("Pharmacie A صيدلية", "Pharmacie A")
