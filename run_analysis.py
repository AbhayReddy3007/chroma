import asyncio
from chroma_main.tools import get_dimension_i_patent_data

async def main():
    results = await asyncio.gather(
        get_dimension_i_patent_data("Axitinib"),
        get_dimension_i_patent_data("Minocycline"),
        return_exceptions=True,
    )

    for r in results:
        print(f"\n{'='*60}")

        # Handle exceptions from asyncio.gather
        if isinstance(r, Exception):
            print(f"ERROR: {r}")
            continue

        drug = r.get("drug_name", "Unknown")
        print(f"Drug:    {drug}")

        # "No PDFs found" early-exit path — missing several keys
        if "error" in r:
            print(f"ERROR:   {r['error']}")
            print(f"Hint:    Check that the GCS folder name exactly matches '{drug}'")
            print(f"         (normalised: '{drug.lower().replace(' ', '').replace('-', '')}')")
            continue

        print(f"Patents: {len(r.get('patents', []))}")
        print(f"Time:    {r.get('processing_time_seconds', 'N/A')}s")
        print(f"Source:  {r.get('phase_data_source', 'N/A')}")
        print(f"Cache:   {'Yes' if r.get('from_cache') else 'No'}")
        print(f"Excel:   {r.get('excel_path', 'N/A')}")

asyncio.run(main())
