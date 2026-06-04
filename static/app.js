// Manejo de Accordion
document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', () => {
        const item = header.parentElement;
        const isActive = item.classList.contains('active');
        
        // Cerrar todos
        document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('active'));
        
        // Abrir si no estaba activo
        if (!isActive) {
            item.classList.add('active');
        }
    });
});

// Referencias del DOM
const loader = document.getElementById('loader');
const imageUpload = document.getElementById('image-upload');
const btnReset = document.getElementById('btn-reset');
const btnMagic = document.getElementById('btn-magic-pipeline');
const imgOriginal = document.getElementById('img-original');
const imgProcesada = document.getElementById('img-procesada');
const imgColormap = document.getElementById('img-colormap');
const colormapContainer = document.getElementById('ccl-colormap-container');

// Métricas DOM
const mArea = document.getElementById('m-area');
const mPerimetro = document.getElementById('m-perimetro');
const mSimetria = document.getElementById('m-simetria');
const mCentroide = document.getElementById('m-centroide');

// Helpers Loader
function showLoader() { loader.style.display = 'flex'; }
function hideLoader() { loader.style.display = 'none'; }

// Upload de Imagen
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    showLoader();
    const formData = new FormData();
    formData.append('image', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        hideLoader();
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Mostrar imágenes y esconder placeholders
        imgOriginal.src = data.original;
        imgOriginal.style.display = 'block';
        imgProcesada.src = data.actual;
        imgProcesada.style.display = 'block';
        
        document.querySelectorAll('.empty-state').forEach(el => el.style.display = 'none');
        
        // Dibujar Histograma
        dibujarHistograma(data.histogram);
        resetMetrics();
    })
    .catch(err => {
        hideLoader();
        console.error(err);
    });
});

// Restablecer Imagen
btnReset.addEventListener('click', () => {
    showLoader();
    fetch('/reset', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        hideLoader();
        imgProcesada.src = data.actual;
        dibujarHistograma(data.histogram);
        resetMetrics();
    })
    .catch(err => {
        hideLoader();
        console.error(err);
    });
});

// Acción Genérica
function applyAction(action, params = {}) {
    showLoader();
    fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, params })
    })
    .then(res => res.json())
    .then(data => {
        hideLoader();
        if (data.error) {
            alert(data.error);
            return;
        }

        imgProcesada.src = data.actual;
        
        if (data.histogram) {
            dibujarHistograma(data.histogram);
        }

        // CCL & Métricas
        if (action === 'analizar' || action === 'pipeline_botella') {
            if (data.colormap) {
                imgColormap.src = data.colormap;
                colormapContainer.style.display = 'block';
            } else {
                colormapContainer.style.display = 'none';
            }

            if (data.metricas) {
                mArea.innerText = data.metricas.Area || '-';
                mPerimetro.innerText = data.metricas.Perimetro || '-';
                mSimetria.innerText = data.metricas["Simetria (IoU)"] || '-';
                mCentroide.innerText = data.metricas.Centroide ? `(${data.metricas.Centroide[0]}, ${data.metricas.Centroide[1]})` : '-';
            }
        } else {
            colormapContainer.style.display = 'none';
            resetMetrics();
        }
    })
    .catch(err => {
        hideLoader();
        console.error(err);
    });
}

function resetMetrics() {
    mArea.innerText = '-';
    mPerimetro.innerText = '-';
    mSimetria.innerText = '-';
    mCentroide.innerText = '-';
    colormapContainer.style.display = 'none';
}

// Pipeline Mágico para la Botella
btnMagic.addEventListener('click', () => {
    const estrategia = document.getElementById('magic-estrategia').value;
    const compacidad_min = document.getElementById('magic-compacidad').value;
    applyAction('pipeline_botella', { estrategia, compacidad_min });
});

// Toggles de Parámetros Dinámicos
function toggleRuidoParams() {
    const tipo = document.getElementById('ruido-tipo').value;
    document.getElementById('param-sal-pimienta').style.display = tipo === 'sal_pimienta' ? 'block' : 'none';
    document.getElementById('param-gaussiano').style.display = tipo === 'gaussiano' ? 'block' : 'none';
}

function toggleFiltroParams() {
    const tipo = document.getElementById('filtro-tipo').value;
    
    document.getElementById('filter-ksize-container').style.display = (tipo === 'nlm') ? 'none' : 'block';
    document.getElementById('params-gauss-sigma').style.display = tipo === 'gaussiano' ? 'block' : 'none';
    document.getElementById('params-bilateral').style.display = tipo === 'bilateral' ? 'block' : 'none';
    document.getElementById('params-nlm').style.display = tipo === 'nlm' ? 'block' : 'none';
    document.getElementById('params-sobel').style.display = tipo === 'sobel' ? 'block' : 'none';
}

function toggleSegParams() {
    const tipo = document.getElementById('seg-tipo').value;
    document.getElementById('param-seg-manual').style.display = tipo === 'manual' ? 'block' : 'none';
    document.getElementById('param-seg-adaptativo').style.display = tipo === 'adaptativo' ? 'block' : 'none';
}

// Eventos de botones con lecturas dinámicas
document.getElementById('btn-apply-noise').addEventListener('click', () => {
    const tipo = document.getElementById('ruido-tipo').value;
    let params = { tipo };
    if (tipo === 'sal_pimienta') {
        params.densidad = document.getElementById('ruido-densidad').value;
    } else {
        params.media = document.getElementById('ruido-media').value;
        params.varianza = document.getElementById('ruido-varianza').value;
    }
    applyAction('ruido', params);
});

document.getElementById('btn-apply-filter').addEventListener('click', () => {
    const tipo = document.getElementById('filtro-tipo').value;
    let params = { tipo };
    params.k = document.getElementById('filtro-k').value;
    
    if (tipo === 'gaussiano') {
        params.sigma = document.getElementById('filtro-sigma').value;
    } else if (tipo === 'bilateral') {
        params.sigma_color = document.getElementById('filtro-sigmacolor').value;
        params.sigma_space = document.getElementById('filtro-sigmaspace').value;
    } else if (tipo === 'nlm') {
        params.h = document.getElementById('filtro-nlm-h').value;
    } else if (tipo === 'sobel') {
        params.eje = document.getElementById('filtro-sobel-eje').value;
    }
    applyAction('filtrar', params);
});

document.getElementById('btn-apply-segment').addEventListener('click', () => {
    const tipo = document.getElementById('seg-tipo').value;
    let params = { tipo };
    if (tipo === 'manual') {
        params.val = document.getElementById('seg-val-manual').value;
    } else if (tipo === 'adaptativo') {
        params.block = document.getElementById('seg-adapt-block').value;
        params.c = document.getElementById('seg-adapt-c').value;
    }
    applyAction('segmentar', params);
});

document.getElementById('btn-apply-morf').addEventListener('click', () => {
    const tipo = document.getElementById('morf-tipo').value;
    const forma = document.getElementById('morf-forma').value;
    const ksize = document.getElementById('morf-ksize').value;
    applyAction('morfologia', { tipo, forma, kw: ksize, kh: ksize });
});

document.getElementById('btn-apply-dct').addEventListener('click', () => {
    const calidad = document.getElementById('dct-calidad').value;
    showLoader();
    fetch('/compress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ calidad })
    })
    .then(res => res.json())
    .then(data => {
        hideLoader();
        if (data.error) {
            alert(data.error);
            return;
        }
        imgProcesada.src = data.comprimida;
        
        // Agregar PSNR al registro de métricas sin borrar el resto
        const psnrDiv = document.createElement('div');
        psnrDiv.className = 'metric-item';
        psnrDiv.innerHTML = `
            <div class="metric-label">PSNR Compresión</div>
            <div class="metric-value" style="color: #ff007f;">${data.psnr} dB</div>
        `;
        // Evitar duplicados
        const oldPsnr = document.getElementById('psnr-temp');
        if (oldPsnr) oldPsnr.remove();
        psnrDiv.id = 'psnr-temp';
        document.getElementById('metricas-container').appendChild(psnrDiv);
    })
    .catch(err => {
        hideLoader();
        console.error(err);
    });
});

// Dibujar Histograma con Plotly
function dibujarHistograma(hist) {
    const trace = {
        x: Array.from({length: 256}, (_, i) => i),
        y: hist,
        type: 'bar',
        marker: {
            color: '#66fcf1'
        }
    };

    const layout = {
        margin: { t: 10, b: 30, l: 45, r: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            color: '#c5c6c7',
            family: 'Outfit'
        },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zeroline: false
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zeroline: false
        }
    };

    const config = { responsive: true, displayModeBar: false };

    Plotly.newPlot('plotly-histogram', [trace], layout, config);
}
