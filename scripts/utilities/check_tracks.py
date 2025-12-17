import xml.etree.ElementTree as ET

tree = ET.parse('outputs/test_optimized_tracks_final.xml')
root = tree.getroot()

video_tracks = root.findall('.//video/track')
print(f'Video tracks: {len(video_tracks)}')

clipitems = root.findall('.//video//clipitem')
print(f'Total clipitems: {len(clipitems)}')

telop_clipitems = [c for c in clipitems if c.find('.//file[@id="file-2"]') is not None]
print(f'Telop clipitems: {len(telop_clipitems)}')
