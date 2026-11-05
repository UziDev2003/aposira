"""Generate actual downloadable reports; formula-safe spreadsheet output."""
from io import BytesIO
import pandas as pd


def export_table(frame, format_type='CSV', title='APOSIRA report'):
    frame=frame.copy()
    for col in frame.select_dtypes(include=['object','string']):
        frame[col]=frame[col].map(lambda v: "'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v)
    if format_type.upper()=='CSV':
        return frame.to_csv(index=False).encode('utf-8-sig'),'text/csv','csv'
    if format_type.upper()=='EXCEL':
        out=BytesIO()
        with pd.ExcelWriter(out,engine='openpyxl') as writer:
            frame.to_excel(writer,index=False,sheet_name='APOSIRA')
        return out.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','xlsx'
    if format_type.upper()=='PDF':
        from fpdf import FPDF
        pdf=FPDF();pdf.set_auto_page_break(auto=True,margin=15);pdf.add_page();pdf.set_font('Helvetica',size=14)
        pdf.multi_cell(0,8,title.encode('latin-1','replace').decode('latin-1'))
        pdf.set_font('Helvetica',size=9)
        for row in frame.to_dict(orient='records'):
            for key,value in row.items():
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0,5,f'{key}: {value}'.encode('latin-1','replace').decode('latin-1'))
            pdf.ln(3)
        return bytes(pdf.output()),'application/pdf','pdf'
    raise ValueError('Supported formats: CSV, Excel, PDF')
