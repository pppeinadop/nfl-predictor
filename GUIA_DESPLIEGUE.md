# Guía paso a paso: publicar el predictor en línea sin pagar

Esta guía utiliza GitHub para guardar el código, Streamlit Community Cloud para publicar la interfaz y MongoDB Atlas para conservar las predicciones. No requiere que la computadora personal permanezca encendida.

## Parte A. Crear la base de datos en MongoDB Atlas

1. Abre MongoDB Atlas y crea una cuenta.
2. Crea un clúster del nivel gratuito disponible.
3. Crea un usuario de base de datos distinto de tu cuenta de Atlas. Usa una contraseña larga y única.
4. En los permisos de ese usuario, limita el acceso a la base `nfl_predictor` si la configuración de Atlas lo permite.
5. En la configuración de red, agrega el acceso requerido por Streamlit Community Cloud. Como el servicio puede no tener una IP de salida fija, la opción `0.0.0.0/0` permite conexiones desde cualquier IP; úsala únicamente junto con contraseña robusta y permisos mínimos.
6. En **Connect**, selecciona el método de conexión de la aplicación y copia la URI `mongodb+srv://...`.
7. Sustituye los datos de ejemplo por el usuario, contraseña y clúster reales. Si la contraseña contiene símbolos como `@`, `:`, `/`, `#` o `%`, codifícalos para que la URI sea válida.
8. No compartas la URI ni la publiques en GitHub.

## Parte B. Preparar GitHub

1. Crea un repositorio nuevo en GitHub.
2. Descarga el ZIP actualizado del proyecto y descomprímelo en tu computadora.
3. Sube los archivos de la carpeta del proyecto a la raíz del repositorio. Deben quedar `app.py`, `requirements.txt`, `config.py` y los módulos Python al mismo nivel.
4. No subas `.env` ni `.streamlit/secrets.toml`. El ZIP contiene `.env.example`, que solo tiene marcadores de posición.
5. Puedes mantener el repositorio privado si la configuración de despliegue te permite conectarlo; si lo haces público, recuerda que el código será visible. Las claves deben permanecer en Secrets, nunca en el código.

## Parte C. Publicar en Streamlit Community Cloud

1. Abre Streamlit Community Cloud e inicia sesión usando la cuenta de GitHub.
2. Selecciona **Create app**.
3. Elige el repositorio, la rama y el archivo principal `app.py`.
4. Antes de terminar, abre la sección **Advanced settings / Secrets** de la aplicación.
5. Pega la configuración TOML, sustituyendo los marcadores por tus valores reales:

   ```toml
   MONGO_URI = "mongodb+srv://USUARIO:CONTRASENA@CLUSTER.mongodb.net/?retryWrites=true&w=majority"
   MONGO_DB_NAME = "nfl_predictor"
   ODDS_API_KEY = "TU_CLAVE_DE_THE_ODDS_API"
   CFBD_API_KEY = "TU_CLAVE_DE_COLLEGE_FOOTBALL_DATA"
   ```

   Si solo usarás NFL, puedes omitir `CFBD_API_KEY`. Si no vas a refrescar cuotas desde la aplicación, puedes omitir temporalmente `ODDS_API_KEY`.

6. Guarda los secretos y despliega la aplicación.
7. Al terminar, Streamlit mostrará una dirección pública parecida a `https://nombre-de-tu-app.streamlit.app`. Esa dirección será tu acceso web; no necesitas comprar un dominio para empezar.

## Parte D. Probar antes de usarlo

1. Abre la URL de la aplicación.
2. Verifica que la barra lateral muestre la base `nfl_predictor` y que no aparezca un error de conexión.
3. En **Generar**, selecciona liga, temporada y semana.
4. Deja activada la opción de usar caché de mercado para no consumir cuota. Si no hay cuotas en caché, el sistema podrá generar predicciones sin los datos de mercado.
5. Selecciona **Oficial (crear una sola vez)** y genera la predicción de una semana futura.
6. Ejecuta nuevamente la opción oficial para esa semana. Los pronósticos ya guardados deben permanecer iguales.
7. Selecciona **Actualizar antes del partido (guardar versión)** para crear una versión adicional de un partido que todavía no haya comenzado. Revisa el historial de actualizaciones en la pestaña de partidos.
8. Para un partido iniciado, el sistema debe bloquear nuevas versiones prepartido.
9. Después de que termine un partido, abre la semana y pulsa **Sincronizar resultados finales**. Esto actualiza los marcadores y la evaluación contra la predicción oficial, sin recalcularla.
10. Activa **Refrescar odds** únicamente cuando quieras consultar The Odds API y aceptes consumir cuota.

## Uso semanal recomendado

- Jueves, aproximadamente a las 8:00 a. m. de Ciudad Juárez: abre la aplicación y genera la predicción oficial de la semana.
- Antes de cada partido que aún no haya comenzado: si hay noticias importantes, genera una actualización prepartido. Esta versión no reemplaza la oficial.
- Después de los partidos: usa **Sincronizar resultados finales** para actualizar resultados y métricas.

## Limitaciones del nivel gratuito

- El alojamiento gratuito puede dormir o suspender aplicaciones que no se utilizan. La primera carga después de un periodo de inactividad puede tardar.
- No se debe asumir que el sistema ejecutará automáticamente una tarea el jueves a las 8:00 a. m. si la aplicación está dormida. El flujo recomendado es abrirla y ejecutar la generación manualmente.
- MongoDB Atlas gratuito tiene límites de almacenamiento, conexiones y recursos. Evita guardar grandes archivos de datos dentro de MongoDB; el proyecto guarda principalmente documentos de predicción, versiones y caché.
- La dirección `streamlit.app` es un subdominio del proveedor, no un dominio personalizado propio.

## Si aparece un error

- **Error de MongoDB:** revisa `MONGO_URI`, usuario, contraseña y la lista de acceso de red de Atlas.
- **Error al instalar paquetes:** revisa el registro del despliegue; Streamlit instala las dependencias de `requirements.txt`.
- **No aparecen cuotas:** comprueba si existe caché. Para actualizarla, activa explícitamente `Refrescar odds`.
- **No se generan predicciones:** revisa que la temporada, semana y liga sean correctas y consulta el registro de salida de la aplicación.
- **La app no actualiza el resultado:** verifica que el proveedor ya publique el marcador final y pulsa `Sincronizar resultados finales`.
