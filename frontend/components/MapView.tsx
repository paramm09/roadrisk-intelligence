"use client";

/**
 * Historical collision map (client-only, loaded via next/dynamic).
 * Shows the DISTRIBUTION of past police-reported collisions from the API -
 * it is NOT a prediction of where future accidents will happen.
 *
 * Basemap: Leaflet + OpenStreetMap (no API key, no CARTO/Mapbox/Google).
 * Markers: Leaflet CircleMarkers on a canvas renderer - lightweight enough for
 * the full (capped) grid-cell set, with circle size = collision count.
 */
import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { MapPoint } from "@/types";

/** Severity colours: Slight = blue, Serious = amber, Fatal = red (legend matches these). */
const COLORS = { Slight: "#38bdf8", Serious: "#f59e0b", Fatal: "#ef4444" } as const;

/** Great Britain, before real data arrives. */
const GB_CENTER: [number, number] = [54.5, -3.0];
const GB_ZOOM = 6;
const OSM_TILES = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

function dominant(p: MapPoint): string {
  if (p.fatal >= Math.max(p.slight, p.serious) * 0.15 && p.fatal > 0) return COLORS.Fatal;
  if (p.serious > p.slight) return COLORS.Serious;
  return COLORS.Slight;
}

/** Circle size communicates the collision count in the cell. */
function radiusFor(total: number): number {
  return Math.min(4 + Math.sqrt(Math.max(total, 1)) * 1.7, 22);
}

function validPoints(points: MapPoint[]): MapPoint[] {
  return points.filter(
    (p) => Number.isFinite(p.lat) && Number.isFinite(p.lon) && p.lat >= -90 && p.lat <= 90,
  );
}

function CollisionLayer({ points }: { points: MapPoint[] }) {
  const map = useMap();

  useEffect(() => {
    const canvas = L.canvas({ padding: 0.5 });
    const group = L.layerGroup();
    const valid = validPoints(points);

    for (const p of valid) {
      const color = dominant(p);
      const marker = L.circleMarker([p.lat, p.lon], {
        renderer: canvas,
        radius: radiusFor(p.total),
        color,
        weight: 1.5,
        fillColor: color,
        fillOpacity: 0.55,
      });
      marker.bindPopup(
        `<div style="min-width:170px">
           <div style="font-weight:700;font-size:12.5px;margin-bottom:4px">${p.total.toLocaleString()} collisions (0.01&deg; cell)</div>
           <div style="font-size:12px;color:#38bdf8">Slight: ${p.slight.toLocaleString()}</div>
           <div style="font-size:12px;color:#f59e0b">Serious: ${p.serious.toLocaleString()}</div>
           <div style="font-size:12px;color:#ef4444">Fatal: ${p.fatal.toLocaleString()}</div>
           <div style="font-size:10.5px;color:#64748b;margin-top:5px">lat ${p.lat.toFixed(4)}, lon ${p.lon.toFixed(4)}</div>
           <div style="font-size:10px;color:#64748b;margin-top:4px">Historical record - not a prediction.</div>
         </div>`,
      );
      group.addLayer(marker);
    }

    group.addTo(map);

    // Fit the view to the real coordinates (with padding); never a forced fixed zoom.
    if (valid.length) {
      const bounds = L.latLngBounds(valid.map((p) => [p.lat, p.lon] as [number, number]));
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 12 });
    }

    return () => {
      map.removeLayer(group);
    };
  }, [map, points]);

  return null;
}

export default function MapView({ points }: { points: MapPoint[] }) {
  const memoPoints = useMemo(() => points, [points]);
  return (
    <MapContainer
      center={GB_CENTER}
      zoom={GB_ZOOM}
      scrollWheelZoom
      className="h-[650px] w-full lg:h-[700px]"
      attributionControl
    >
      <TileLayer attribution={OSM_ATTRIBUTION} url={OSM_TILES} maxZoom={19} />
      <CollisionLayer points={memoPoints} />
    </MapContainer>
  );
}
