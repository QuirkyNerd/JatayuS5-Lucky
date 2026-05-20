import { jsPDF } from 'jspdf';

const COLORS = {
  primary: [99, 102, 241],
  dark: [15, 23, 42],
  mid: [51, 65, 85],
  muted: [100, 116, 139],
  light: [248, 250, 252],
  border: [203, 213, 225],
  danger: [239, 68, 68],
  success: [16, 185, 129],
  warning: [245, 158, 11],
};

function safeStr(val, fallback = '—') {
  if (val === null || val === undefined || val === '') return fallback;
  return String(val);
}

function truncate(str, maxLen = 800) {
  if (!str) return '—';
  return str.length > maxLen ? str.slice(0, maxLen) + ' [truncated]' : str;
}

function riskColor(score) {
  if (score >= 70) return COLORS.danger;
  if (score >= 40) return COLORS.warning;
  return COLORS.success;
}

export function generatePdf(c) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = doc.internal.pageSize.getWidth();
  const H = doc.internal.pageSize.getHeight();
  const M = 18;
  const RW = W - M * 2;
  let y = M;

  const newPage = () => {
    doc.addPage();
    y = M;
  };
  const checkY = (needed = 12) => {
    if (y + needed > H - 15) newPage();
  };

  const setFont = (style = 'normal', size = 10) => {
    doc.setFont('helvetica', style);
    doc.setFontSize(size);
  };

  const setColor = (rgb) => doc.setTextColor(...rgb);

  const line = (color = COLORS.border) => {
    doc.setDrawColor(...color);
    doc.setLineWidth(0.3);
    doc.line(M, y, W - M, y);
    y += 4;
  };

  const sectionTitle = (title) => {
    checkY(14);
    y += 3;
    doc.setFillColor(...COLORS.primary);
    doc.roundedRect(M, y, RW, 8, 1.5, 1.5, 'F');
    setFont('bold', 9);
    setColor([255, 255, 255]);
    doc.text(title.toUpperCase(), M + 3, y + 5.5);
    y += 12;
    setColor(COLORS.dark);
  };

  const labelValue = (label, value, opts = {}) => {
    const { bold = false, color = null } = opts;
    checkY(8);
    setFont('bold', 8.5);
    setColor(COLORS.muted);
    doc.text(`${label}:`, M, y);
    const labelW = doc.getTextWidth(`${label}: `);
    setFont(bold ? 'bold' : 'normal', 8.5);
    setColor(color || COLORS.dark);
    doc.text(safeStr(value), M + labelW + 1, y);
    y += 6;
  };

  const multiLine = (text, maxW = RW, fontSize = 8.5) => {
    setFont('normal', fontSize);
    setColor(COLORS.mid);
    const lines = doc.splitTextToSize(text, maxW);
    lines.forEach(ln => {
      checkY(6);
      doc.text(ln, M, y);
      y += 5.5;
    });
    y += 2;
  };

  doc.setFillColor(...COLORS.dark);
  doc.rect(0, 0, W, 28, 'F');

  setFont('bold', 18);
  setColor([255, 255, 255]);
  doc.text('CodePerfect Audit', M, 13);

  setFont('normal', 9);
  setColor([148, 163, 184]);
  doc.text('Clinical Coding Audit Report', M, 20);

  const caseLabel = `Case #${c.id}`;
  setFont('bold', 9);
  setColor([255, 255, 255]);
  const badgeW = doc.getTextWidth(caseLabel) + 10;
  doc.setFillColor(...COLORS.primary);
  doc.roundedRect(W - M - badgeW, 6, badgeW, 10, 2, 2, 'F');
  doc.text(caseLabel, W - M - badgeW + 5, 12.5);

  y = 33;

  setFont('normal', 8);
  setColor(COLORS.muted);
  const dateStr = c.created_at ? new Date(c.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) : '—';
  const nowStr = new Date().toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  doc.text(`Audit Date: ${dateStr}`, M, y);
  doc.text(`Generated: ${nowStr}`, M + 70, y);
  y += 4;
  line(COLORS.border);

  const metrics = [
    { label: 'Risk Score', value: `${(c.risk_score || 0).toFixed(0)}%`, color: riskColor(c.risk_score || 0) },
    { label: 'Revenue Impact', value: `$${(c.revenue_impact || 0).toFixed(0)}`, color: COLORS.primary },
    { label: 'Accuracy', value: `${(c.coding_accuracy || 0).toFixed(1)}%`, color: COLORS.success },
    { label: 'Processing', value: `${(c.processing_time || 0).toFixed(2)}s`, color: COLORS.muted },
  ];
  const boxW = RW / metrics.length;
  metrics.forEach((m, i) => {
    const bx = M + i * boxW;
    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(...COLORS.border);
    doc.setLineWidth(0.3);
    doc.roundedRect(bx, y, boxW - 2, 16, 2, 2, 'FD');
    setFont('bold', 12);
    setColor(m.color);
    doc.text(m.value, bx + (boxW - 2) / 2, y + 9, { align: 'center' });
    setFont('normal', 7);
    setColor(COLORS.muted);
    doc.text(m.label, bx + (boxW - 2) / 2, y + 14, { align: 'center' });
  });
  y += 22;

  const statusColors = { pending: COLORS.warning, reviewed: COLORS.primary, approved: COLORS.success, rejected: COLORS.danger };
  const sc = statusColors[c.status] || COLORS.muted;
  setFont('bold', 8);
  setColor(sc);
  doc.setFillColor(...sc.map(v => Math.min(v + 180, 255)));
  const statusText = `Status: ${(c.status || 'unknown').toUpperCase()}`;
  const sW = doc.getTextWidth(statusText) + 8;
  doc.roundedRect(M, y, sW, 7, 1.5, 1.5, 'F');
  setColor(sc);
  doc.text(statusText, M + 4, y + 5);
  y += 12;

  sectionTitle('Clinical Summary');
  multiLine(truncate(c.summary || 'No summary available.', 600));

  sectionTitle('Patient / Encounter Note');
  multiLine(truncate(c.input_text || '—', 1000));

  sectionTitle('AI-Generated Codes');
  const aiCodes = Array.isArray(c.ai_codes) ? c.ai_codes : [];
  if (aiCodes.length === 0) {
    multiLine('No codes generated.');
  } else {
    checkY(8);
    doc.setFillColor(...COLORS.primary.map(v => Math.min(v + 140, 255)));
    doc.rect(M, y, RW, 7, 'F');
    setFont('bold', 8);
    setColor(COLORS.primary);
    doc.text('Code', M + 2, y + 5);
    doc.text('Description', M + 28, y + 5);
    doc.text('Confidence', W - M - 20, y + 5, { align: 'right' });
    y += 7;

    aiCodes.forEach((item, i) => {
      checkY(7);
      if (i % 2 === 0) {
        doc.setFillColor(248, 250, 252);
        doc.rect(M, y, RW, 7, 'F');
      }
      setFont('bold', 8);
      setColor(COLORS.primary);
      doc.text(safeStr(item.code), M + 2, y + 5);
      setFont('normal', 8);
      setColor(COLORS.mid);
      const desc = safeStr(item.description || item.desc || '—');
      doc.text(desc.slice(0, 55), M + 28, y + 5);
      const conf = item.confidence != null ? `${(item.confidence * 100).toFixed(0)}%` : '—';
      setFont('bold', 8);
      setColor((item.confidence || 0) >= 0.8 ? COLORS.success : COLORS.warning);
      doc.text(conf, W - M - 2, y + 5, { align: 'right' });
      y += 7;
    });
    y += 3;
  }

  sectionTitle('Clinician-Assigned Codes');
  const humanCodes = Array.isArray(c.human_codes) ? c.human_codes : [];
  if (humanCodes.length === 0) {
    multiLine('No human codes recorded.');
  } else {
    humanCodes.forEach(code => {
      checkY(6);
      setFont('normal', 8.5);
      setColor(COLORS.dark);
      doc.text(`• ${safeStr(code)}`, M + 2, y);
      y += 6;
    });
    y += 2;
  }

  sectionTitle('Discrepancy Analysis');
  const discrepancies = Array.isArray(c.discrepancies) ? c.discrepancies : [];
  const missed = discrepancies.filter(d => d.type === 'missed' || d.issue === 'missed').length;
  const unsupported = discrepancies.filter(d => d.type === 'unsupported' || d.issue === 'unsupported').length;

  labelValue('Total Discrepancies', discrepancies.length, { bold: true, color: discrepancies.length > 0 ? COLORS.danger : COLORS.success });
  labelValue('Missed Codes', missed, { color: missed > 0 ? COLORS.warning : COLORS.success });
  labelValue('Unsupported Codes', unsupported, { color: unsupported > 0 ? COLORS.danger : COLORS.success });

  if (discrepancies.length > 0) {
    y += 2;
    discrepancies.slice(0, 6).forEach(d => {
      checkY(7);
      setFont('italic', 8);
      setColor(COLORS.muted);
      const entry = d.code ? `${d.code} – ${d.issue || d.type || ''}` : safeStr(JSON.stringify(d)).slice(0, 80);
      doc.text(`• ${entry}`, M + 3, y);
      y += 6;
    });
  }

  const totalPages = doc.internal.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    doc.setFillColor(...COLORS.dark);
    doc.rect(0, H - 10, W, 10, 'F');
    setFont('normal', 7);
    setColor([148, 163, 184]);
    doc.text('CodePerfect Audit – Confidential Clinical Report', M, H - 4);
    doc.text(`Page ${p} of ${totalPages}`, W - M, H - 4, { align: 'right' });
  }

  doc.save(`case_report_${c.id}.pdf`);
}
