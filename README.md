# Predictor NFL / NCAAF

Aplicación en Streamlit para generar predicciones previas al partido, consultar las predicciones guardadas y comparar la predicción oficial con el marcador final. Usa MongoDB para conservar la información aunque la aplicación esté alojada en la nube.

## Reglas de conservación de predicciones

- **Predicción oficial:** se guarda una sola vez por partido y no se reemplaza con ejecuciones posteriores.
- **Actualización prepartido:** se guarda como una versión separada en `prediction_versions`; no modifica el registro oficial.
- **Bloqueo por kickoff:** si el partido ya comenzó o no se puede determinar con seguridad su hora de inicio, no se permite crear una nueva predicción oficial ni una actualización prepartido.
- **Resultado final:** cuando se actualizan los datos de un partido ya guardado, solo se sincronizan los campos del resultado y las métricas se calculan usando la predicción oficial original.
- **Sin simulación en vivo:** no se generan pronósticos nuevos después del inicio del partido.
- **Cuotas:** por defecto se utiliza la caché de MongoDB. Solo se consume cuota de The Odds API cuando se solicita explícitamente `--refresh-odds` o se activa “Refrescar odds” en la aplicación. Si no hay caché, el sistema sigue generando la predicción y deja vacíos los campos de mercado.
- **Datos temporales:** los promedios del modelo usan partidos de semanas anteriores, no estadísticas del partido que se está pronosticando.

## Archivos principales

| Archivo | Función |
|---|---|
| `app.py` | Interfaz web Streamlit |
| `main.py` | Generación desde terminal |
| `rebuild_weeks.py` | Flujo de generación y manejo de caché de cuotas |
| `predict_games.py` | Construcción de predicciones NFL/NCAAF |
| `elo_model.py` | Ratings Elo |
| `team_stats.py` | Estadísticas históricas por equipo |
| `adjustments.py` | Ajustes por lesiones, descanso, rivalidad y clima |
| `data_fetch.py` | Calendario NFL, estadísticas y cuotas |
| `ncaaf_fetch.py` | Calendario y datos NCAAF |
| `player_props.py` | Proyecciones opcionales de estadísticas individuales |
| `db.py` | MongoDB, predicciones oficiales, versiones y caché |
| `report.py`, `generate_report.py`, `view_week.py` | Reportes y consulta de semanas guardadas |
| `config.py` | Parámetros y configuración del sistema |

Los antiguos scripts `analyze_*` y `sweep_*` se retiraron del paquete principal: eran herramientas experimentales de calibración, no necesarias para ejecutar la aplicación. También se retiró `enrich_with_odds.py` porque podía modificar registros existentes de predicciones y no respetaba el nuevo bloqueo de la predicción oficial.

## Ejecución local

1. Instala Python 3.11 o una versión compatible con las dependencias.
2. Crea un entorno virtual e instala dependencias:

   ```bash
   python -m venv .venv
   ```

   En Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copia `.env.example` como `.env` y agrega las claves que vayas a utilizar.
4. Inicia MongoDB local o configura `MONGO_URI` para conectarte a una base remota.
5. Ejecuta la interfaz:

   ```bash
   streamlit run app.py
   ```

### Generar desde terminal

Predicción oficial (no consume cuota de odds si no existe caché):

```bash
python main.py --season 2026 --week 6 --league nfl
```

Crear una actualización prepartido como versión separada:

```bash
python main.py --season 2026 --week 6 --league nfl --mode update
```

Actualizar cuotas de manera explícita (consume cuota de The Odds API):

```bash
python main.py --season 2026 --week 6 --league nfl --refresh-odds
```

En la interfaz web, selecciona la liga, temporada y semana; elige si se trata de la predicción oficial o una actualización prepartido. La opción de refrescar cuotas está separada y debe activarse solo cuando se quiera gastar cuota.

## Publicar sin costo con Streamlit Community Cloud + MongoDB Atlas

### 1. Crear la base de datos remota

1. Crea una cuenta gratuita en MongoDB Atlas y crea un clúster gratuito disponible en tu región.
2. Crea un usuario de base de datos con contraseña fuerte y permisos de lectura/escritura únicamente en la base `nfl_predictor`.
3. En Network Access, autoriza las conexiones necesarias. Streamlit Community Cloud no garantiza una IP de salida fija; para una aplicación pública sencilla normalmente se configura `0.0.0.0/0`, por lo que es importante usar una contraseña fuerte y permisos mínimos.
4. Copia la cadena de conexión `mongodb+srv://...` de Atlas. Sustituye usuario, contraseña y nombre de base correctamente; codifica los caracteres especiales de la contraseña en la URI.

### 2. Subir el proyecto a GitHub

1. Crea un repositorio en GitHub.
2. Sube los archivos del proyecto, incluido `requirements.txt` y `README.md`.
3. **No subas `.env`, `secrets.toml`, claves de API ni contraseñas.** El archivo `.gitignore` ya excluye los secretos locales.
4. Para un primer despliegue, verifica que el repositorio tenga el archivo `app.py` en la raíz.

### 3. Publicar la aplicación

1. Entra a Streamlit Community Cloud e inicia sesión con GitHub.
2. Crea una aplicación nueva y selecciona el repositorio, la rama y `app.py` como archivo principal.
3. En la configuración de secretos de la aplicación, agrega los valores necesarios con formato TOML, por ejemplo:

   ```toml
   MONGO_URI = "mongodb+srv://USUARIO:CONTRASENA@CLUSTER.mongodb.net/?retryWrites=true&w=majority"
   MONGO_DB_NAME = "nfl_predictor"
   ODDS_API_KEY = "TU_CLAVE_DE_ODDS_API"
   CFBD_API_KEY = "TU_CLAVE_DE_CFBD"
   ```

   Si solo utilizarás NFL, `CFBD_API_KEY` puede omitirse. Si no deseas consultar cuotas desde la web, `ODDS_API_KEY` puede omitirse hasta que lo necesites.

4. Inicia el despliegue. Streamlit instalará los paquetes de `requirements.txt` y te asignará una dirección web pública `*.streamlit.app`.
5. Abre la dirección y verifica la conexión a MongoDB, la selección de temporada y liga, y la lectura de predicciones guardadas.

### 4. Comprobaciones antes de usarla

- Genera una predicción oficial para un partido futuro y verifica que quede guardada.
- Ejecuta de nuevo el proceso oficial: el registro original debe permanecer sin cambios.
- Crea una actualización prepartido: debe aparecer como una versión independiente.
- Prueba con un partido que ya haya comenzado: debe bloquearse cualquier predicción nueva.
- Tras un partido finalizado, pulsa **Sincronizar resultados finales** para actualizar el marcador y comparar contra la predicción oficial.
- Verifica el panel de uso de The Odds API antes de marcar “Refrescar odds”.

## Límites del alojamiento gratuito

El alojamiento gratuito puede suspender aplicaciones inactivas y tener límites de recursos. Por ello, la aplicación está diseñada para ejecutarse cuando el usuario la abre, no para mantener procesos permanentes ni tareas programadas. La generación de los jueves a las 8:00 a. m. debe hacerse manualmente desde la web; el alojamiento no garantiza ejecutar tareas a esa hora si la aplicación está dormida.

## Variables de entorno

- `MONGO_URI`: cadena de conexión a MongoDB.
- `MONGO_DB_NAME`: nombre de la base de datos (`nfl_predictor` por defecto).
- `ODDS_API_KEY`: clave opcional para The Odds API.
- `CFBD_API_KEY`: clave opcional para datos NCAAF.
