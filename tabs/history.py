import streamlit as st
import pandas as pd
import os

if 'page' not in st.session_state:
    st.session_state.page = 1

# Number of pages
total_pages = 3

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

def next_page():
    if st.session_state.page < total_pages:
        st.session_state.page += 1

def set_page(page):
    st.session_state.page

def history():
    st.title("History")
       
    if os.path.exists('predictions.csv'):
        history_data = pd.read_csv('predictions.csv')
        
        page_size = 5
        total_rows = history_data.shape[0]
        num_pages = total_rows // page_size + 1

        # Menampilkan tabel berdasarkan halaman yang dipilih
        page_number = st.number_input('Halaman Tabel', min_value=1, max_value=num_pages, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        if start_idx < end_idx:
            # Reset index agar dimulai dari nomor 1
            history_data_display = history_data.iloc[start_idx:end_idx].reset_index(drop=True)
            history_data_display.index += start_idx + 1  # Mengubah index agar dimulai dari nomor yang sesuai

            st.markdown("**Tabel History Prediksi**")
            st.dataframe(history_data_display)

            # Opsi untuk menghapus entri riwayat
            st.subheader("Hapus History")
            delete_index = st.selectbox("Pilih nomor entri untuk dihapus", options=history_data_display.index.tolist())
            if st.button('Hapus Entri', key='delete_entry'):
                if len(history_data_display) == 1:  # Jika hanya ada satu baris tersisa
                    delete_index = 0  # Hapus baris pertama karena hanya ada satu baris
                else:
                    delete_index -= start_idx + 1  # Adjust delete_index to match original dataframe

                # Menghapus entri yang dipilih
                deleted_entry = history_data.iloc[delete_index].to_frame().transpose()
                history_data.drop(history_data.index[delete_index], inplace=True)
                history_data.reset_index(drop=True, inplace=True)  # Reset index setelah penghapusan
                history_data.to_csv('predictions.csv', index=False)

                # Simpan entri yang dihapus ke dalam file sementara
                if os.path.exists('deleted_entries.csv'):
                    deleted_entries = pd.read_csv('deleted_entries.csv')
                    deleted_entries = pd.concat([deleted_entries, deleted_entry], ignore_index=True)
                else:
                    deleted_entries = deleted_entry.copy()
                
                deleted_entries.to_csv('deleted_entries.csv', index=False)

                st.success('Entri history berhasil dihapus.')
                st.experimental_rerun()

            # Tombol untuk mengembalikan entri yang dihapus sebelumnya
            if os.path.exists('deleted_entries.csv'):
                if st.button('Undo', key='undo_entry'):
                    deleted_entries = pd.read_csv('deleted_entries.csv')
                    last_deleted_entry = deleted_entries.tail(1)

                    # Masukkan kembali entri yang dihapus ke dalam 'predictions.csv'
                    history_data = pd.concat([history_data, last_deleted_entry], ignore_index=True)
                    history_data.to_csv('predictions.csv', index=False)

                    # Hapus entri terakhir dari 'deleted_entries.csv'
                    deleted_entries = deleted_entries.iloc[:-1, :]
                    deleted_entries.to_csv('deleted_entries.csv', index=False)

                    st.success('Entri berhasil dikembalikan.')
                    st.experimental_rerun()
        else:
            st.write("Belum ada data untuk ditampilkan.")

    else:
        st.write("Belum ada data prediksi yang disimpan.")

    # Tombol untuk menghapus riwayat
    if st.button('Hapus Semua History', key='delete_all'):
        if os.path.exists('predictions.csv'):
            os.remove('predictions.csv')
            st.success('History prediksi berhasil dihapus.')
            st.experimental_rerun()
        else:
            st.warning('Tidak ada data History yang ditemukan untuk dihapus.')
