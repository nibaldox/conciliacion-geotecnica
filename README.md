# Conciliación Geotécnica 3D

Herramienta para la conciliación geométrica de taludes de mina, comparando superficies de Diseño vs As-Built (Topografía/Escáner).

## Características

*   **Carga de Mallas 3D:** Soporte para formatos STL, OBJ, PLY y DXF (3DFACES y Polilíneas).
*   **Generación de Secciones:**
    *   Manual: Click en vista de planta.
    *   Automática (Línea): Definición de línea de cresta y espaciamiento.
    *   Automática (DXF): Carga de ejes desde archivo DXF.
    *   **Azimut Inteligente:** Cálculo automático de la dirección de corte perpendicular a la cara del talud (descendente).
*   **Análisis Geométrico (Algoritmo RDP):**
    *   Simplificación precisa de perfiles para detección robusta de Crestas y Patas.
    *   Extracción automática de Altura de Banco, Ángulo de Cara, Ancho de Berma.
    *   Comparación Diseño vs Real basada en emparejamiento por cota (elevación).
*   **Reportabilidad:**
    *   **Excel:** Tablas detalladas de cumplimiento, desviaciones y dashboard de KPIs.
    *   **Word:** Informe técnico generado automáticamente con gráficos de perfil por sección.
    *   **Imágenes ZIP:** Exportación masiva de gráficos de secciones.
*   **Interfaz Web (Streamlit):** Visualización 3D interactiva, configuración de tolerancias en tiempo real.
*   **CLI:** Interfaz de línea de comandos para procesamiento por lotes.

## Instalación

1.  Clonar repositorio:
    ```bash
    git clone https://github.com/nibaldox/conciliacion-geotecnica.git
    cd conciliacion-geotecnica
    ```

2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Interfaz Web (App)
Para iniciar la aplicación visual:
```bash
streamlit run app.py
```
Acceder en el navegador a `http://localhost:8501`.

### Línea de Comandos (CLI)
Para procesar archivos sin interfaz gráfica:
```bash
python cli.py --design path/to/design.stl --topo path/to/scan.stl --length 150 --spacing 20 --output ./resultados
```
Opciones completas del CLI:
*   `--design`: Malla de diseño.
*   `--topo`: Malla as-built.
*   `--sections`: Archivo JSON con definiciones (opcional) o `dxf_poly`.
*   `--dxf_poly`: Archivo DXF con polilínea eje.
*   `--report`: Generar reporte Word adicional al Excel.

## Estructura de Salida (Excel)
El archivo Excel generado incluye:
1.  **Resumen:** Configuración, tolerancias y estadísticas globales.
2.  **Bancos:** Tabla detallada fila por fila de cada banco conciliado (H, Ang, Berma).
3.  **Inter-Rampa:** Comparación de ángulos globales e inter-rampa.
4.  **Dashboard:** Resumen visual de semáforo (CUMPLE / FUERA TOL / NO CUMPLE).

## Notas Técnicas
*   **Unidades:** El sistema asume que las mallas están en **Metros**.
*   **Coordenadas:** Se asume sistema de coordenadas compatible entre diseño y topo.
*   **DXF:** Para exportar desde Vulcan/Minesight, asegurar exportar como "3D Faces" o "Triangulaciones" para superficies, y "Polylines" para ejes.
