import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import {
  MapPin,
  UserSearch,
  Phone,
  Mail,
  Loader2,
  Map as MapIcon,
  Star,
  Globe,
} from "lucide-react";

const DoctorSearchPage = ({ defaultSpecialty = "" }) => {
  const [location, setLocation] = useState("");
  const [specialty, setSpecialty] = useState(defaultSpecialty);
  const [doctors, setDoctors] = useState([]);
  const [coords, setCoords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [geoError, setGeoError] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [map, setMap] = useState(null);
  const [markers, setMarkers] = useState([]);
  const mapRef = useRef(null);
  const doctorsPerPage = 6;
  
  // Google Maps API key - you'll need to set this
  const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "YOUR_GOOGLE_MAPS_API_KEY";

  const paginatedDoctors = doctors.slice(
    (currentPage - 1) * doctorsPerPage,
    currentPage * doctorsPerPage
  );

  // Load Google Maps script
  useEffect(() => {
    // Check if API key is properly set
    if (GOOGLE_MAPS_API_KEY === "YOUR_GOOGLE_MAPS_API_KEY" || !GOOGLE_MAPS_API_KEY) {
      console.error("Google Maps API key not configured. Please set VITE_GOOGLE_MAPS_API_KEY in your .env file.");
      setGeoError(true);
      return;
    }

    const loadGoogleMaps = () => {
      // Check if Google Maps is already loaded
      if (window.google && window.google.maps) {
        initializeMap();
        return;
      }
      
      // Check if script is already being loaded
      const existingScript = document.querySelector('script[src*="maps.googleapis.com"]');
      if (existingScript) {
        existingScript.addEventListener('load', initializeMap);
        return;
      }
      
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_API_KEY}&libraries=places&loading=async`;
      script.async = true;
      script.defer = true;
      script.onload = initializeMap;
      script.onerror = () => {
        console.error('Failed to load Google Maps API');
        setGeoError(true);
      };
      document.head.appendChild(script);
    };
    
    loadGoogleMaps();
    
    // Cleanup function
    return () => {
      clearMarkers();
    };
  }, [GOOGLE_MAPS_API_KEY]);
  
  // Initialize Google Map
  const initializeMap = () => {
    if (mapRef.current && window.google) {
      const mapInstance = new window.google.maps.Map(mapRef.current, {
        center: { lat: 28.6139, lng: 77.2090 }, // Default to Delhi
        zoom: 12,
        styles: [
          {
            featureType: "poi.medical",
            elementType: "geometry",
            stylers: [{ color: "#ffeaa7" }]
          }
        ]
      });
      setMap(mapInstance);
      
      // Try to get user's current location
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const pos = {
              lat: position.coords.latitude,
              lng: position.coords.longitude,
            };
            setCoords(pos);
            mapInstance.setCenter(pos);
            
            // Reverse geocode to get location name
            const geocoder = new window.google.maps.Geocoder();
            geocoder.geocode({ location: pos }, (results, status) => {
              if (status === "OK" && results[0]) {
                const addressComponents = results[0].address_components;
                const city = addressComponents.find(component => 
                  component.types.includes("locality") || 
                  component.types.includes("administrative_area_level_2")
                );
                setLocation(city ? city.long_name : results[0].formatted_address);
              }
            });
          },
          () => {
            setGeoError(true);
          }
        );
      }
    }
  };

  // Clear existing markers
  const clearMarkers = () => {
    if (markers && markers.length > 0) {
      markers.forEach(marker => {
        if (marker && marker.setMap) {
          marker.setMap(null);
        }
      });
      setMarkers([]);
    }
  };
  
  // Add markers to map
  const addMarkersToMap = (doctorsList) => {
    if (!map || !window.google || !doctorsList || doctorsList.length === 0) return;
    
    clearMarkers();
    const newMarkers = [];
    
    doctorsList.forEach((doctor, index) => {
      // Validate doctor data
      if (!doctor.lat || !doctor.lng || isNaN(doctor.lat) || isNaN(doctor.lng)) {
        console.warn(`Invalid coordinates for doctor: ${doctor.name}`);
        return;
      }

      try {
        const marker = new window.google.maps.Marker({
          position: { lat: parseFloat(doctor.lat), lng: parseFloat(doctor.lng) },
          map: map,
          title: doctor.name,
          icon: {
            url: 'https://maps.google.com/mapfiles/ms/icons/red-dot.png',
            scaledSize: new window.google.maps.Size(32, 32)
          }
        });
        
        const infoWindow = new window.google.maps.InfoWindow({
          content: `
            <div style="padding: 10px; max-width: 250px;">
              <h3 style="margin: 0 0 5px 0; color: #333;">${doctor.name || 'Unknown'}</h3>
              <p style="margin: 2px 0; color: #666;"><strong>Specialty:</strong> ${doctor.specialty || 'General'}</p>
              <p style="margin: 2px 0; color: #666;"><strong>Phone:</strong> ${doctor.phone || 'Not available'}</p>
              ${doctor.rating ? `<p style="margin: 2px 0; color: #666;"><strong>Rating:</strong> ${doctor.rating}/5 ⭐</p>` : ''}
              <p style="margin: 5px 0 0 0; font-size: 12px; color: #888;">${doctor.location || 'Location not available'}</p>
            </div>
          `
        });
        
        marker.addListener('click', () => {
          infoWindow.open(map, marker);
        });
        
        newMarkers.push(marker);
      } catch (error) {
        console.error(`Error creating marker for doctor: ${doctor.name}`, error);
      }
    });
    
    setMarkers(newMarkers);
    
    // Fit map to show all markers
    if (newMarkers.length > 0) {
      try {
        const bounds = new window.google.maps.LatLngBounds();
        newMarkers.forEach(marker => bounds.extend(marker.getPosition()));
        map.fitBounds(bounds);
        
        // Set a reasonable zoom level if there's only one marker
        if (newMarkers.length === 1) {
          setTimeout(() => {
            if (map.getZoom() > 15) {
              map.setZoom(15);
            }
          }, 100);
        }
      } catch (error) {
        console.error('Error fitting map bounds:', error);
      }
    }
  };

  const handleSearch = async () => {
    if (!location) {
      alert("Please enter a location.");
      return; 
    }
    setLoading(true);
    setDoctors([]);
    clearMarkers();
    
    try {
      // Fetch doctors from backend using Google Places API
      const res = await axios.get("http://127.0.0.1:8000/api/search-doctors", {
        params: { location, specialty },
      });
      
      const doctorsList = res.data;
      setDoctors(doctorsList);
      
      if (doctorsList.length > 0) {
        // Center map on first result
        const firstDoctor = doctorsList[0];
        const centerCoords = { lat: firstDoctor.lat, lng: firstDoctor.lng };
        setCoords(centerCoords);
        
        if (map) {
          map.setCenter(centerCoords);
          addMarkersToMap(doctorsList);
        }
      } else {
        alert("No doctors found in this location. Try a different search.");
      }
    } catch (err) {
      console.error("Error fetching doctors:", err);
      if (err.response?.status === 500 && err.response?.data?.detail?.includes("API key")) {
        alert("Google Maps API key not configured. Please contact administrator.");
      } else {
        alert("Failed to fetch doctors. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 bg-white dark:bg-black rounded-lg shadow-lg">
      <h1 className="text-3xl font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
        <UserSearch size={28} /> Find Doctors Near You
      </h1>

      <div className="flex flex-wrap gap-4 items-center">
        <div className="relative w-full sm:w-1/3">
          <MapPin className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Enter location"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <div className="relative w-full sm:w-1/3">
          <UserSearch className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Specialty (e.g. Cardiologist)"
            value={specialty}
            onChange={(e) => setSpecialty(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <Button
          onClick={handleSearch}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 font-semibold"
        >
          {loading && <Loader2 className="animate-spin" size={18} />}
          Search
        </Button>
      </div>

      {geoError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600 font-medium">
            {GOOGLE_MAPS_API_KEY === "YOUR_GOOGLE_MAPS_API_KEY" || !GOOGLE_MAPS_API_KEY 
              ? "Google Maps API key not configured. Please contact administrator." 
              : "Could not get your location automatically. Please enter manually."}
          </p>
        </div>
      )}

      {/* Google Map */}
      <div className="relative">
        <div 
          ref={mapRef} 
          className="h-[400px] w-full rounded-xl shadow-md"
          style={{ minHeight: '400px' }}
        />
        {!map && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-xl">
            <p className="text-gray-500 italic flex items-center gap-2">
              <MapIcon size={18} /> Loading Google Maps...
            </p>
          </div>
        )}
      </div>

      {/* List of doctors */}
      <div className="grid gap-6 pt-6">
        {!loading && doctors.length === 0 && (
          <p className="text-center text-gray-500 italic">No doctors found.</p>
        )}
        {paginatedDoctors.map((doc, idx) => (
          <div
            key={idx}
            className="border border-gray-300 dark:border-gray-700 p-5 rounded-xl shadow-sm hover:shadow-lg transition-shadow duration-300"
          >
            <div className="flex justify-between items-start mb-2">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <UserSearch size={20} /> {doc.name}
              </h2>
              {doc.rating > 0 && (
                <div className="flex items-center gap-1 bg-green-100 px-2 py-1 rounded-full">
                  <Star size={14} className="text-yellow-500 fill-current" />
                  <span className="text-sm font-medium text-green-800">{doc.rating}</span>
                </div>
              )}
            </div>
            <p className="text-gray-600 dark:text-gray-300 mb-1">
              <strong>Specialty: </strong> {doc.specialty}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-1 flex items-center gap-2">
              <MapPin size={16} /> {doc.location}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-2 flex items-center gap-2">
              <Phone size={16} /> {doc.phone}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex items-center gap-2 px-4 py-2"
                onClick={() => window.open(`tel:${doc.phone}`, "_blank")}
                disabled={doc.phone === "Not available"}
              >
                <Phone size={16} /> Call
              </Button>
              {doc.website && (
                <Button
                  variant="outline"
                  className="flex items-center gap-2 px-4 py-2"
                  onClick={() => window.open(doc.website, "_blank")}
                >
                  <Globe size={16} /> Website
                </Button>
              )}
            </div>
          </div>
        ))}
        {doctors.length > doctorsPerPage && (
  <div className="flex justify-center gap-2 pt-4">
    <Button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)}>Prev</Button>
    <Button disabled={currentPage * doctorsPerPage >= doctors.length} onClick={() => setCurrentPage(p => p + 1)}>Next</Button>
  </div>
)}
      </div>
    </div>
  );
};

export default DoctorSearchPage;
