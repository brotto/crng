#!/usr/bin/env python3
"""
Open-Meteo Weather API Client
==============================
Wrapper completo para a API gratuita do Open-Meteo.
Uso nao-comercial, sem API key necessaria.

Endpoints:
  - Forecast:   api.open-meteo.com/v1/forecast (ate 16 dias)
  - Archive:    archive-api.open-meteo.com/v1/archive (historico)

Docs: https://open-meteo.com/en/docs
Free: < 10.000 calls/dia, sem autenticacao

Uso:
  python weather_api.py --current                    # Condicoes atuais SP
  python weather_api.py --hourly --days 1            # Hora-a-hora hoje
  python weather_api.py --daily --days 7             # Proximos 7 dias
  python weather_api.py --archive 2026-04-01 2026-04-05  # Historico
  python weather_api.py --compare 12                 # Compara forecast vs archive para hora X
  python weather_api.py --full                       # Dump completo (todas as variaveis)
  python weather_api.py --city nyc --current         # Outra cidade
  python weather_api.py --lat -22.9 --lon -43.17 --current  # Coordenadas customizadas
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta

# ============================================================
# CIDADES PRE-CONFIGURADAS
# ============================================================

CITIES = {
    "sp":      {"name": "Sao Paulo",   "lat": -23.55, "lon": -46.63,  "tz": "America/Sao_Paulo"},
    "nyc":     {"name": "New York",    "lat":  40.71, "lon": -74.01,  "tz": "America/New_York"},
    "london":  {"name": "London",      "lat":  51.51, "lon":  -0.13,  "tz": "Europe/London"},
    "tokyo":   {"name": "Tokyo",       "lat":  35.68, "lon": 139.69,  "tz": "Asia/Tokyo"},
    "paris":   {"name": "Paris",       "lat":  48.86, "lon":   2.35,  "tz": "Europe/Paris"},
    "berlin":  {"name": "Berlin",      "lat":  52.52, "lon":  13.41,  "tz": "Europe/Berlin"},
    "sydney":  {"name": "Sydney",      "lat": -33.87, "lon": 151.21,  "tz": "Australia/Sydney"},
    "mumbai":  {"name": "Mumbai",      "lat":  19.08, "lon":  72.88,  "tz": "Asia/Kolkata"},
    "rio":     {"name": "Rio de Janeiro", "lat": -22.91, "lon": -43.17, "tz": "America/Sao_Paulo"},
    "bsb":     {"name": "Brasilia",    "lat": -15.79, "lon": -47.88,  "tz": "America/Sao_Paulo"},
}

# ============================================================
# VARIAVEIS DISPONIVEIS (referencia completa)
# ============================================================

HOURLY_VARS = {
    # Temperatura & Umidade
    "temperature_2m":           "Temperatura a 2m (C)",
    "relative_humidity_2m":     "Umidade relativa (%)",
    "dew_point_2m":             "Ponto de orvalho (C)",
    "apparent_temperature":     "Sensacao termica (C)",
    "vapour_pressure_deficit":  "Deficit de pressao de vapor (kPa)",

    # Pressao
    "pressure_msl":             "Pressao ao nivel do mar (hPa)",
    "surface_pressure":         "Pressao de superficie (hPa)",

    # Nuvens
    "cloud_cover":              "Cobertura total de nuvens (%)",
    "cloud_cover_low":          "Nuvens baixas (%)",
    "cloud_cover_mid":          "Nuvens medias (%)",
    "cloud_cover_high":         "Nuvens altas (%)",

    # Vento
    "wind_speed_10m":           "Vento a 10m (km/h)",
    "wind_speed_80m":           "Vento a 80m (km/h)",
    "wind_speed_120m":          "Vento a 120m (km/h)",
    "wind_speed_180m":          "Vento a 180m (km/h)",
    "wind_direction_10m":       "Direcao do vento 10m (graus)",
    "wind_direction_80m":       "Direcao do vento 80m (graus)",
    "wind_direction_120m":      "Direcao do vento 120m (graus)",
    "wind_direction_180m":      "Direcao do vento 180m (graus)",
    "wind_gusts_10m":           "Rajadas de vento (km/h)",

    # Radiacao Solar
    "shortwave_radiation":      "Radiacao de onda curta (W/m2)",
    "direct_radiation":         "Radiacao direta (W/m2)",
    "direct_normal_irradiance": "Irradiancia normal direta (W/m2)",
    "diffuse_radiation":        "Radiacao difusa (W/m2)",
    "global_tilted_irradiance": "Irradiancia global inclinada (W/m2)",

    # Precipitacao
    "precipitation":            "Precipitacao total (mm)",
    "precipitation_probability":"Probabilidade de chuva (%)",
    "rain":                     "Chuva (mm)",
    "showers":                  "Pancadas (mm)",
    "snowfall":                 "Neve (cm)",
    "snow_depth":               "Profundidade da neve (m)",

    # Atmosfera
    "cape":                     "CAPE - energia convectiva (J/kg)",
    "freezing_level_height":    "Altitude do nivel de congelamento (m)",
    "visibility":               "Visibilidade (m)",
    "weather_code":             "Codigo WMO do tempo",
    "is_day":                   "E dia? (0/1)",

    # Evapotranspiracao
    "evapotranspiration":       "Evapotranspiracao (mm)",
    "et0_fao_evapotranspiration": "ET0 FAO (mm)",

    # Solo
    "soil_temperature_0cm":     "Temp. solo 0cm (C)",
    "soil_temperature_6cm":     "Temp. solo 6cm (C)",
    "soil_temperature_18cm":    "Temp. solo 18cm (C)",
    "soil_temperature_54cm":    "Temp. solo 54cm (C)",
    "soil_moisture_0_to_1cm":   "Umidade solo 0-1cm (m3/m3)",
    "soil_moisture_1_to_3cm":   "Umidade solo 1-3cm (m3/m3)",
    "soil_moisture_3_to_9cm":   "Umidade solo 3-9cm (m3/m3)",
    "soil_moisture_9_to_27cm":  "Umidade solo 9-27cm (m3/m3)",
    "soil_moisture_27_to_81cm": "Umidade solo 27-81cm (m3/m3)",
}

DAILY_VARS = {
    "weather_code":                 "Codigo WMO do tempo",
    "temperature_2m_max":           "Temperatura maxima (C)",
    "temperature_2m_mean":          "Temperatura media (C)",
    "temperature_2m_min":           "Temperatura minima (C)",
    "apparent_temperature_max":     "Sensacao termica max (C)",
    "apparent_temperature_mean":    "Sensacao termica media (C)",
    "apparent_temperature_min":     "Sensacao termica min (C)",
    "precipitation_sum":            "Precipitacao total (mm)",
    "rain_sum":                     "Chuva total (mm)",
    "showers_sum":                  "Pancadas total (mm)",
    "snowfall_sum":                 "Neve total (cm)",
    "precipitation_hours":          "Horas com precipitacao",
    "precipitation_probability_max":"P(chuva) maxima (%)",
    "precipitation_probability_mean":"P(chuva) media (%)",
    "precipitation_probability_min":"P(chuva) minima (%)",
    "sunrise":                      "Nascer do sol",
    "sunset":                       "Por do sol",
    "sunshine_duration":            "Duracao do sol (s)",
    "daylight_duration":            "Duracao da luz do dia (s)",
    "wind_speed_10m_max":           "Vento maximo (km/h)",
    "wind_gusts_10m_max":           "Rajada maxima (km/h)",
    "wind_direction_10m_dominant":  "Direcao dominante (graus)",
    "shortwave_radiation_sum":      "Radiacao solar total (MJ/m2)",
    "et0_fao_evapotranspiration":   "ET0 FAO (mm)",
    "uv_index_max":                 "Indice UV maximo",
    "uv_index_clear_sky_max":       "Indice UV max ceu limpo",
}

CURRENT_VARS = {
    "temperature_2m":        "Temperatura (C)",
    "relative_humidity_2m":  "Umidade relativa (%)",
    "apparent_temperature":  "Sensacao termica (C)",
    "is_day":                "E dia? (0/1)",
    "precipitation":         "Precipitacao (mm)",
    "rain":                  "Chuva (mm)",
    "showers":               "Pancadas (mm)",
    "snowfall":              "Neve (cm)",
    "weather_code":          "Codigo WMO",
    "cloud_cover":           "Nuvens (%)",
    "pressure_msl":          "Pressao mar (hPa)",
    "surface_pressure":      "Pressao superficie (hPa)",
    "wind_speed_10m":        "Vento (km/h)",
    "wind_direction_10m":    "Direcao vento (graus)",
    "wind_gusts_10m":        "Rajadas (km/h)",
}

# WMO Weather codes
WMO_CODES = {
    0: "Ceu limpo", 1: "Parcialmente limpo", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Nevoeiro", 48: "Nevoeiro com geada",
    51: "Garoa leve", 53: "Garoa moderada", 55: "Garoa densa",
    56: "Garoa congelante leve", 57: "Garoa congelante densa",
    61: "Chuva leve", 63: "Chuva moderada", 65: "Chuva forte",
    66: "Chuva congelante leve", 67: "Chuva congelante forte",
    71: "Neve leve", 73: "Neve moderada", 75: "Neve forte",
    77: "Graos de neve", 80: "Pancadas leves", 81: "Pancadas moderadas",
    82: "Pancadas violentas", 85: "Neve leve (pancadas)",
    86: "Neve forte (pancadas)", 95: "Trovoada", 96: "Trovoada com granizo leve",
    99: "Trovoada com granizo forte",
}

# Presets de variaveis para consultas rapidas
PRESETS = {
    "basic": {
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                   "cloud_cover", "wind_speed_10m", "weather_code"],
        "daily":  ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                   "sunrise", "sunset"],
    },
    "crng_cast": {
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                   "pressure_msl", "cloud_cover", "cloud_cover_low",
                   "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m",
                   "wind_gusts_10m", "precipitation", "rain", "showers",
                   "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                   "cape", "visibility", "weather_code", "is_day",
                   "soil_temperature_0cm", "soil_moisture_0_to_1cm",
                   "vapour_pressure_deficit"],
        "daily":  ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                   "sunshine_duration", "shortwave_radiation_sum", "sunrise", "sunset",
                   "wind_speed_10m_max", "wind_gusts_10m_max"],
    },
    "solar": {
        "hourly": ["shortwave_radiation", "direct_radiation",
                   "direct_normal_irradiance", "diffuse_radiation",
                   "global_tilted_irradiance", "is_day", "cloud_cover"],
        "daily":  ["shortwave_radiation_sum", "sunshine_duration",
                   "sunrise", "sunset", "uv_index_max"],
    },
    "wind": {
        "hourly": ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m",
                   "wind_direction_10m", "wind_direction_80m",
                   "wind_gusts_10m", "pressure_msl"],
        "daily":  ["wind_speed_10m_max", "wind_gusts_10m_max",
                   "wind_direction_10m_dominant"],
    },
    "soil": {
        "hourly": ["soil_temperature_0cm", "soil_temperature_6cm",
                   "soil_temperature_18cm", "soil_temperature_54cm",
                   "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
                   "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
                   "soil_moisture_27_to_81cm"],
        "daily":  [],
    },
    "full": {
        "hourly": list(HOURLY_VARS.keys()),
        "daily":  list(DAILY_VARS.keys()),
    },
}

# Variaveis disponiveis no archive (subconjunto do forecast)
ARCHIVE_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "vapour_pressure_deficit",
    "pressure_msl", "surface_pressure",
    "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "shortwave_radiation", "direct_radiation", "direct_normal_irradiance",
    "diffuse_radiation", "global_tilted_irradiance", "sunshine_duration",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
    "wind_direction_100m", "wind_gusts_10m",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    "et0_fao_evapotranspiration", "weather_code",
]

ARCHIVE_DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "sunshine_duration", "daylight_duration",
    "sunrise", "sunset", "weather_code",
]


# ============================================================
# API CLIENT
# ============================================================

class OpenMeteoClient:
    """Cliente para a API Open-Meteo."""

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, city="sp", lat=None, lon=None, tz=None):
        if lat is not None and lon is not None:
            self.lat = lat
            self.lon = lon
            self.tz = tz or "GMT"
            self.city_name = f"({lat}, {lon})"
        elif city in CITIES:
            c = CITIES[city]
            self.lat = c["lat"]
            self.lon = c["lon"]
            self.tz = c["tz"]
            self.city_name = c["name"]
        else:
            raise ValueError(f"Cidade '{city}' nao encontrada. Disponiveis: {list(CITIES.keys())}")

    def _fetch(self, url, params):
        """Faz request GET e retorna JSON."""
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        full_url = f"{url}?{query}"
        try:
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": True, "reason": str(e), "url": full_url}

    def current(self):
        """Condicoes atuais."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.tz,
            "current": ",".join(CURRENT_VARS.keys()),
        }
        return self._fetch(self.FORECAST_URL, params)

    def hourly(self, variables=None, days=1, past_days=0, preset="basic"):
        """Forecast horario."""
        if variables is None:
            variables = PRESETS.get(preset, PRESETS["basic"])["hourly"]
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.tz,
            "hourly": ",".join(variables),
            "forecast_days": days,
            "past_days": past_days,
        }
        return self._fetch(self.FORECAST_URL, params)

    def daily(self, variables=None, days=7, preset="basic"):
        """Forecast diario."""
        if variables is None:
            variables = PRESETS.get(preset, PRESETS["basic"])["daily"]
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.tz,
            "daily": ",".join(variables),
            "forecast_days": days,
        }
        return self._fetch(self.FORECAST_URL, params)

    def archive(self, start_date, end_date, hourly=None, daily=None):
        """Dados historicos (observacoes reais de estacoes)."""
        if hourly is None:
            hourly = ["temperature_2m", "relative_humidity_2m", "precipitation",
                      "cloud_cover", "wind_speed_10m", "pressure_msl",
                      "shortwave_radiation", "weather_code"]
        if daily is None:
            daily = ["temperature_2m_max", "temperature_2m_min",
                     "precipitation_sum", "sunrise", "sunset"]

        # Filter to only vars available in archive
        hourly = [v for v in hourly if v in ARCHIVE_HOURLY_VARS]
        daily  = [v for v in daily if v in ARCHIVE_DAILY_VARS]

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.tz,
            "start_date": start_date,
            "end_date": end_date,
        }
        if hourly:
            params["hourly"] = ",".join(hourly)
        if daily:
            params["daily"] = ",".join(daily)

        return self._fetch(self.ARCHIVE_URL, params)

    def archive_full(self, start_date, end_date):
        """Archive com TODAS as variaveis disponiveis."""
        return self.archive(start_date, end_date,
                           hourly=ARCHIVE_HOURLY_VARS,
                           daily=ARCHIVE_DAILY_VARS)

    def compare_forecast_vs_archive(self, date, hour=12):
        """Compara forecast vs observacao real para uma data/hora."""
        # Forecast
        fcast = self.hourly(
            variables=PRESETS["crng_cast"]["hourly"],
            days=1
        )
        # Archive
        arch = self.archive(date, date,
                           hourly=["temperature_2m", "relative_humidity_2m",
                                   "precipitation", "cloud_cover", "wind_speed_10m",
                                   "pressure_msl", "weather_code"])
        return {"forecast": fcast, "archive": arch, "target_hour": hour}


# ============================================================
# FORMATTERS
# ============================================================

def format_wmo(code):
    """Converte codigo WMO para descricao."""
    if code is None:
        return "-"
    return WMO_CODES.get(int(code), f"Codigo {code}")

def print_current(data, city_name):
    """Imprime condicoes atuais formatadas."""
    if "error" in data:
        print(f"ERRO: {data.get('reason', 'desconhecido')}")
        return

    c = data.get("current", {})
    print(f"\n{'='*50}")
    print(f"  CONDICOES ATUAIS — {city_name}")
    print(f"{'='*50}")
    print(f"  Hora:          {c.get('time', '?')}")
    print(f"  Temperatura:   {c.get('temperature_2m', '?')}°C")
    print(f"  Sensacao:      {c.get('apparent_temperature', '?')}°C")
    print(f"  Umidade:       {c.get('relative_humidity_2m', '?')}%")
    print(f"  Nuvens:        {c.get('cloud_cover', '?')}%")
    print(f"  Vento:         {c.get('wind_speed_10m', '?')} km/h")
    print(f"  Dir. vento:    {c.get('wind_direction_10m', '?')}°")
    print(f"  Rajadas:       {c.get('wind_gusts_10m', '?')} km/h")
    print(f"  Precipitacao:  {c.get('precipitation', '?')} mm")
    print(f"  Pressao (mar): {c.get('pressure_msl', '?')} hPa")
    print(f"  Tempo:         {format_wmo(c.get('weather_code'))}")
    print(f"  E dia:         {'Sim' if c.get('is_day') else 'Nao'}")
    print(f"{'='*50}\n")

def print_hourly(data, city_name, variables=None):
    """Imprime dados horarios formatados."""
    if "error" in data:
        print(f"ERRO: {data.get('reason', 'desconhecido')}")
        return

    h = data.get("hourly", {})
    times = h.get("time", [])
    if not times:
        print("Sem dados horarios.")
        return

    # Default: show basic vars
    show_vars = ["temperature_2m", "relative_humidity_2m", "precipitation",
                 "cloud_cover", "wind_speed_10m", "weather_code"]
    show_vars = [v for v in show_vars if v in h]

    print(f"\n{'='*80}")
    print(f"  DADOS HORARIOS — {city_name}")
    print(f"{'='*80}")

    # Header
    header = f"{'Hora':<18}"
    for v in show_vars:
        label = v.replace("temperature_2m", "Temp").replace("relative_humidity_2m", "Humid") \
                 .replace("precipitation", "Precip").replace("cloud_cover", "Cloud") \
                 .replace("wind_speed_10m", "Vento").replace("weather_code", "Tempo")
        header += f"{label:>8}"
    print(header)
    print("-" * len(header))

    for i in range(len(times)):
        row = f"{times[i]:<18}"
        for v in show_vars:
            val = h[v][i]
            if v == "weather_code":
                row += f"{'  ' + format_wmo(val)[:6]:>8}"
            elif val is None:
                row += f"{'—':>8}"
            elif isinstance(val, float):
                row += f"{val:>7.1f} "
            else:
                row += f"{val:>7} "
        print(row)

    print(f"{'='*80}\n")

def print_daily(data, city_name):
    """Imprime dados diarios formatados."""
    if "error" in data:
        print(f"ERRO: {data.get('reason', 'desconhecido')}")
        return

    d = data.get("daily", {})
    times = d.get("time", [])
    if not times:
        print("Sem dados diarios.")
        return

    print(f"\n{'='*75}")
    print(f"  PREVISAO DIARIA — {city_name}")
    print(f"{'='*75}")
    print(f"{'Data':<12} {'Max':>6} {'Min':>6} {'Precip':>7} {'Nascer':>8} {'Por':>8}")
    print("-" * 55)

    for i in range(len(times)):
        date = times[i]
        tmax = d.get("temperature_2m_max", [None]*(i+1))[i]
        tmin = d.get("temperature_2m_min", [None]*(i+1))[i]
        prec = d.get("precipitation_sum", [None]*(i+1))[i]
        sunrise = d.get("sunrise", [None]*(i+1))[i]
        sunset  = d.get("sunset", [None]*(i+1))[i]

        tmax_s = f"{tmax:.1f}°C" if tmax is not None else "—"
        tmin_s = f"{tmin:.1f}°C" if tmin is not None else "—"
        prec_s = f"{prec:.1f}mm" if prec is not None else "—"
        rise_s = sunrise.split("T")[1] if sunrise else "—"
        set_s  = sunset.split("T")[1] if sunset else "—"

        print(f"{date:<12} {tmax_s:>6} {tmin_s:>6} {prec_s:>7} {rise_s:>8} {set_s:>8}")

    print(f"{'='*75}\n")

def print_comparison(result, city_name):
    """Imprime comparacao forecast vs archive."""
    fcast = result["forecast"]
    arch  = result["archive"]
    hour  = result["target_hour"]

    if "error" in fcast or "error" in arch:
        print(f"ERRO: forecast={fcast.get('error')}, archive={arch.get('error')}")
        return

    fh = fcast.get("hourly", {})
    ah = arch.get("hourly", {})

    # Find the target hour index
    target_str = f"{hour:02d}:00"
    f_idx = None
    a_idx = None

    for i, t in enumerate(fh.get("time", [])):
        if target_str in t:
            f_idx = i
            break
    for i, t in enumerate(ah.get("time", [])):
        if target_str in t:
            a_idx = i
            break

    print(f"\n{'='*55}")
    print(f"  FORECAST vs OBSERVADO — {city_name} {hour:02d}:00")
    print(f"{'='*55}")

    if f_idx is not None and a_idx is not None:
        f_temp = fh["temperature_2m"][f_idx]
        a_temp = ah["temperature_2m"][a_idx]
        erro = abs(f_temp - a_temp) if f_temp and a_temp else None

        print(f"  Forecast:   {f_temp}°C" if f_temp else "  Forecast:   (sem dado)")
        print(f"  Observado:  {a_temp}°C" if a_temp else "  Observado:  (sem dado)")
        if erro is not None:
            print(f"  Erro:       {erro:.1f}°C")
    else:
        print(f"  Hora {hour:02d}:00 nao encontrada nos dados.")

    print(f"{'='*55}\n")

def print_vars():
    """Lista todas as variaveis disponiveis."""
    print(f"\n{'='*70}")
    print(f"  VARIAVEIS HORARIAS (forecast)")
    print(f"{'='*70}")
    for var, desc in HOURLY_VARS.items():
        archive_flag = " [A]" if var in ARCHIVE_HOURLY_VARS else ""
        print(f"  {var:<35} {desc}{archive_flag}")

    print(f"\n{'='*70}")
    print(f"  VARIAVEIS DIARIAS (forecast)")
    print(f"{'='*70}")
    for var, desc in DAILY_VARS.items():
        archive_flag = " [A]" if var in ARCHIVE_DAILY_VARS else ""
        print(f"  {var:<40} {desc}{archive_flag}")

    print(f"\n{'='*70}")
    print(f"  VARIAVEIS CURRENT")
    print(f"{'='*70}")
    for var, desc in CURRENT_VARS.items():
        print(f"  {var:<30} {desc}")

    print(f"\n{'='*70}")
    print(f"  PRESETS")
    print(f"{'='*70}")
    for name, preset in PRESETS.items():
        n_h = len(preset["hourly"])
        n_d = len(preset["daily"])
        print(f"  {name:<15} {n_h} hourly vars, {n_d} daily vars")

    print(f"\n  [A] = disponivel tambem no Archive (dados historicos)\n")

def export_json(data, filename):
    """Salva resultado em JSON."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Dados salvos em: {filename}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Open-Meteo Weather API Client — CRNG-Cast Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s --current                          # SP agora
  %(prog)s --city nyc --current               # NYC agora
  %(prog)s --hourly --days 2                  # SP proximas 48h
  %(prog)s --hourly --preset crng_cast        # Todas vars para CRNG-Cast
  %(prog)s --daily --days 7                   # SP proximos 7 dias
  %(prog)s --archive 2026-04-01 2026-04-05    # SP historico
  %(prog)s --archive 2026-04-01 2026-04-05 --full  # Historico completo
  %(prog)s --compare 12                       # Forecast vs real 12h
  %(prog)s --vars                             # Lista todas variaveis
  %(prog)s --lat -22.9 --lon -43.17 --current # Coordenadas custom
  %(prog)s --json output.json --current       # Salva JSON
        """
    )

    # Location
    parser.add_argument("--city", default="sp", help=f"Cidade: {', '.join(CITIES.keys())}")
    parser.add_argument("--lat", type=float, help="Latitude customizada")
    parser.add_argument("--lon", type=float, help="Longitude customizada")

    # Modes
    parser.add_argument("--current", action="store_true", help="Condicoes atuais")
    parser.add_argument("--hourly", action="store_true", help="Dados horarios")
    parser.add_argument("--daily", action="store_true", help="Dados diarios")
    parser.add_argument("--archive", nargs=2, metavar=("START", "END"), help="Historico: START END (yyyy-mm-dd)")
    parser.add_argument("--compare", type=int, metavar="HOUR", help="Compara forecast vs archive para hora")
    parser.add_argument("--vars", action="store_true", help="Lista todas variaveis")

    # Options
    parser.add_argument("--days", type=int, default=1, help="Dias de forecast (default: 1)")
    parser.add_argument("--past", type=int, default=0, help="Dias passados para incluir no hourly")
    parser.add_argument("--preset", default="basic",
                       help=f"Preset de variaveis: {', '.join(PRESETS.keys())}")
    parser.add_argument("--full", action="store_true", help="Todas variaveis disponiveis")
    parser.add_argument("--json", metavar="FILE", help="Salvar resultado em arquivo JSON")
    parser.add_argument("--raw", action="store_true", help="Mostra JSON cru")

    args = parser.parse_args()

    # Init client
    if args.lat is not None and args.lon is not None:
        client = OpenMeteoClient(lat=args.lat, lon=args.lon)
    else:
        client = OpenMeteoClient(city=args.city)

    preset = "full" if args.full else args.preset

    # Execute
    if args.vars:
        print_vars()
        return

    if args.current:
        data = client.current()
        if args.raw:
            print(json.dumps(data, indent=2))
        else:
            print_current(data, client.city_name)
        if args.json:
            export_json(data, args.json)
        return

    if args.hourly:
        data = client.hourly(days=args.days, past_days=args.past, preset=preset)
        if args.raw:
            print(json.dumps(data, indent=2))
        else:
            print_hourly(data, client.city_name)
        if args.json:
            export_json(data, args.json)
        return

    if args.daily:
        data = client.daily(days=args.days, preset=preset)
        if args.raw:
            print(json.dumps(data, indent=2))
        else:
            print_daily(data, client.city_name)
        if args.json:
            export_json(data, args.json)
        return

    if args.archive:
        start, end = args.archive
        if args.full:
            data = client.archive_full(start, end)
        else:
            data = client.archive(start, end)
        if args.raw:
            print(json.dumps(data, indent=2))
        else:
            print_hourly(data, f"{client.city_name} [ARCHIVE]")
        if args.json:
            export_json(data, args.json)
        return

    if args.compare is not None:
        today = datetime.now().strftime("%Y-%m-%d")
        data = client.compare_forecast_vs_archive(today, args.compare)
        if args.raw:
            print(json.dumps(data, indent=2))
        else:
            print_comparison(data, client.city_name)
        if args.json:
            export_json(data, args.json)
        return

    # Default: show current
    data = client.current()
    print_current(data, client.city_name)


if __name__ == "__main__":
    main()
