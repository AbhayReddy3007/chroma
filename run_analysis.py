# run_analysis.py
import asyncio
from chroma_main.tools import get_dimension_i_patent_data

async def main():
    results = await asyncio.gather(
        get_dimension_i_patent_data("Axitinib"),
        get_dimension_i_patent_data("Minocycline"),
    )
    for r in results:
        print(f"\n{'='*60}")
        print(f"Drug: {r['drug_name']}")
        print(f"Patents: {len(r['patents'])}")
        print(f"Time: {r['processing_time_seconds']}s")
        print(f"Excel: {r.get('excel_path')}")

asyncio.run(main())
